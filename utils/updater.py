# utils/updater.py

import logging
import math
import json
import redis
import os

from utils.buffer import Buffer
from utils.vlm_api import VLM_API
from common.redis_client import GROUP_INFO


class Updater:
    def __init__(self, config):
        self.config = config
        updater_config = self.config.get("updater", {})
        self.enable_cost_estimate = updater_config.get("enable_cost_estimate", False)
        self.enable_reward_estimate = updater_config.get("enable_reward_estimate", False)
        
        # 从配置中读取权重参数
        priority_config = config.get("priority", {})
        self.w_urgency = priority_config.get("w_urgency", 1.0)
        self.w_scope = priority_config.get("w_scope", 1.0)
        self.w_cost = priority_config.get("w_cost", -1.0)
        self.w_reward = priority_config.get("w_reward", 1.0)
        self.w_dependency = priority_config.get("w_dependency", 3.0)
        
        # Set up VLM
        self.prompt_updater = config.get("prompt", {}).get("updater", {})
        
        config_vlm = config.get("vlm", {}).get("updater", {})
        model_name = config_vlm.get("model", "qwen/qwen2.5-vl-72b-instruct")
        server = config_vlm.get("server", "openrouter")
        base_url = config_vlm.get("base_url", None)
        api_key = config_vlm.get("api_key", None)
        
        self.vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
        
        # Set up Buffer
        self.buffer = Buffer()
        
        # 存储当前优先级最高的问题
        self.highest_priority_question = None
        self.highest_priority_score = float('-inf')


    def add_question(self, question):
        # Finishing Module调用该方法加入一个问题
        self.highest_priority_question = None
        self.highest_priority_score = float('-inf')
        
        # 先计算cost和reward，然后计算dependency，最后status由buffer中的calculate_status()方法生成
        original_questions = self.buffer.get_pending_and_ready_questions()
        all_questions = original_questions.copy()
        all_questions.append(question)
        
        # 为所有问题计算reward
        if self.enable_reward_estimate:
            all_questions = self._get_reward_estimate(all_questions)
        else:
            for q in all_questions:
                q["reward_estimate"] = 0.0
        
        for q in all_questions:
            q["cost_estimate"] = self._get_cost_estimate(q) if self.enable_cost_estimate else 0.0
        
        if len(all_questions) >= 2:
            qid = question["id"]
            new_dependency = self._get_new_dependency(all_questions, qid)
            # new_dependency:
            # {
            #     "depends_on": ["id_1", "id_2", ...],
            #     "required_by": ["id_3", "id_4", ...]
            # }
            
            all_questions = self._merge_dependencies(all_questions, new_dependency, qid)
        
        # 将更新后的问题列表写回buffer
        self.buffer.write_latest_questions(all_questions)
        
        # 重建DAG和计算状态
        self.buffer.build_dag()
        self.buffer.calculate_status()
        
        # 更新优先级分数
        self._update_priority_scores()
    
    
    def complete_question(self, question):
        # Stopping Module调用该方法表示已经完成一个问题
        self.highest_priority_question = None
        self.highest_priority_score = float('-inf')
        
        question_in_progress = self.buffer.get_question_by_id(question["id"])
        if question_in_progress["status"] != "in_progress":
            raise ValueError(f"Question {question_in_progress['id']} is not in progress. Cannot possibly be completed.")
        self.buffer.set_status(question_in_progress["id"], "completed")
        
        # 移除所有和该问题相关的依赖
        other_questions = self.buffer.get_pending_and_ready_questions()
        completed_id = question_in_progress["id"]
        
        # 遍历每个问题，移除对已完成问题的依赖
        for other_question in other_questions:
            if completed_id in other_question["dependency"]:
                other_question["dependency"].remove(completed_id)
        
        # 重新计算reward和cost
        if self.enable_reward_estimate:
            other_questions = self._get_reward_estimate(other_questions)
        else:
            for q in other_questions:
                q["reward_estimate"] = 0.0
        
        for q in other_questions:
            q["cost_estimate"] = self._get_cost_estimate(q) if self.enable_cost_estimate else 0.0
        
        # 将更新后的问题列表写回buffer
        self.buffer.write_latest_questions(other_questions)
        
        # 重建DAG和计算状态
        self.buffer.build_dag()
        self.buffer.calculate_status()

        # 更新优先级分数
        self._update_priority_scores()
    
    
    def answer_question(self, question):
        question_completed = self.buffer.get_question_by_id(question["id"])
        if question_completed["status"] != "completed":
            raise ValueError(f"Question {question_completed['id']} is not completed. Cannot possibly be answered.")
        self.buffer.set_answer(question_completed["id"], question["answer"])
        self.buffer.set_status(question_completed["id"], "answered")
    
    
    def add_answered_question_directly(self, question):
        """
        直接添加一个已回答的问题到缓冲区中，跳过Finishing Module的处理。
        该方法用于处理从Finishing Module直接进入Answering Module的问题。
        """
        question["status"] = "answered"
        question.setdefault("max_steps", 0)
        question.setdefault("used_steps", 0)
        
        # 确保time字段存在
        if "time" not in question:
            question["time"] = {}
        
        self.buffer.add_question(question)
    
    
    def get_highest_priority_question(self):
        """获取当前优先级最高的问题"""
        if self.highest_priority_question is None:
            logging.info(f"[{os.getpid()}](QUE) No pending or ready questions available.")
            return None
        
        return self.highest_priority_question
    
    
    def is_group_completed(self, redis_conn, group_id):
        """
        检查当前问题组是否已全部完成：若 self.buffer 中所有问题均为 answered 状态，
        且问题总数与 GROUP_INFO 中 "num_questions_init" 与 "num_questions_follow_up" 之和一致，则返回 True；否则返回 False。
        """
        buffer_questions = self.buffer.get_buffer()
        if not buffer_questions:
            return False

        # 检查所有问题状态
        all_answered = all(q.get("status") == "answered" for q in buffer_questions)
        if not all_answered:
            return False

        # 获取Redis中该组的题目数量
        num_init = int(redis_conn.get(f"{GROUP_INFO['num_questions_init']}{group_id}") or 0)
        num_follow_up = int(redis_conn.get(f"{GROUP_INFO['num_questions_follow_up']}{group_id}") or 0)
        total_expected = num_init + num_follow_up

        # 检查数量是否一致
        if len(buffer_questions) == total_expected:
            return True
        return False
    
    
    def clear_buffer(self):
        self.buffer.clear()
        self.highest_priority_question = None
        self.highest_priority_score = float('-inf')
        logging.info(f"[{os.getpid()}](QUE) Buffer cleared.")
    
    
    def get_question_by_id(self, question_id):
        return self.buffer.get_question_by_id(question_id)
    
    
    def set_status(self, question_id, status):
        return self.buffer.set_status(question_id, status)
    
    
    def _calculate_priority_score(self, question):
        """根据联合优化公式计算问题的优先级分数"""
        # 提取必要的参数
        urgency = question["urgency"]
        scope_type = question["scope_type"]
        cost = question["cost_estimate"]
        reward = question["reward_estimate"]
        status = question["status"]
        
        # 计算Scope_q: 0 if global, 1 if local
        scope_score = 1 if scope_type == "local" else 0
        
        # 计算Dep_q: 0 if pending, 1 if ready
        dep_score = 1 if status == "ready" else 0
        
        # 应用联合优化公式
        # Score_q = W_urg * (-ln(1 - Urg_q)) + W_scope * Scope_q + W_cost * Cost_q + W_reward * Reward_q + W_dep * Dep_q
        urgency_term = -math.log(1 - urgency + 1e-10)  # 添加一个小值避免log(0)
        
        score = (self.w_urgency * urgency_term + 
                self.w_scope * scope_score + 
                self.w_cost * cost + 
                self.w_reward * reward + 
                self.w_dependency * dep_score)
        
        return score
    
    
    def _update_priority_scores(self):
        """计算所有待处理和就绪问题的优先级分数，并找出分数最高的问题"""
        questions = self.buffer.get_pending_and_ready_questions()
        
        # 重置最高优先级
        self.highest_priority_question = None
        self.highest_priority_score = float('-inf')
        
        for question in questions:
            score = self._calculate_priority_score(question)
            
            # 更新分数最高的问题
            if score > self.highest_priority_score:
                self.highest_priority_score = score
                self.highest_priority_question = question
                
        logging.info(f"更新优先级完成，最高优先级问题ID: {self.highest_priority_question['id'] if self.highest_priority_question else 'None'}, 分数: {self.highest_priority_score}")


    def _get_reward_estimate(self, questions):
        """
        计算每个问题的奖励估计值
        
        Args:
            questions: 问题列表
        
        Returns:
            list: 更新了reward_estimate的问题列表
        """
        # 如果只有一个问题，reward直接设为1
        if len(questions) <= 1:
            for q in questions:
                q["reward_estimate"] = 1.0
            return questions
        
        # 获取prompt模板
        prompt_get_reward = self.prompt_updater.get("get_reward_estimate", "")
        if not prompt_get_reward:
            logging.warning("未找到奖励估计的提示词模板")
            # 如果没有模板，默认所有问题reward为1
            for q in questions:
                q["reward_estimate"] = 1.0
            return questions
        
        # 准备问题数据
        questions_data = []
        for q in questions:
            questions_data.append({
                "id": q["id"],
                "description": q["description"]
            })
        
        # 准备问题的格式化字符串
        questions_str = json.dumps(questions_data, ensure_ascii=False, indent=2)
        
        # 填充提示词模板
        prompt = prompt_get_reward.format(questions=questions_str)
        
        # 发送请求到VLM API
        response = self.vlm.request_with_retry(image=None, prompt=prompt)[0]
        
        # 解析响应获取reward值
        try:
            # 寻找JSON格式的响应
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                reward_data = json.loads(json_str)
                
                # 确保返回格式正确
                if not isinstance(reward_data, dict):
                    logging.error(f"奖励估计格式错误: {reward_data}")
                    # 默认所有问题reward为1
                    for q in questions:
                        q["reward_estimate"] = 1.0
                    return questions
                    
                # 更新问题的reward_estimate
                for q in questions:
                    q_id = q["id"]
                    if q_id in reward_data:
                        # 确保reward是一个数值并在合理范围内
                        reward = float(reward_data[q_id])
                        reward = max(1.0, min(len(questions), reward))  # 限制在1到问题数量之间
                        q["reward_estimate"] = reward
                    else:
                        # 未找到对应ID的reward，设为默认值1
                        q["reward_estimate"] = 1.0
                        logging.warning(f"未找到问题ID {q_id} 的reward值，设为默认值1")
                
                return questions
            else:
                logging.error(f"无法从响应中提取JSON: {response}")
                # 默认所有问题reward为1
                for q in questions:
                    q["reward_estimate"] = 1.0
                return questions
        except Exception as e:
            logging.error(f"解析奖励估计时出错: {e}, 响应: {response}")
            # 发生错误时，默认所有问题reward为1
            for q in questions:
                q["reward_estimate"] = 1.0
            return questions
    
    
    def _get_cost_estimate(self, question):
        return 0.0
    
    
    def _get_new_dependency(self, all_questions, qid):
        # FIXME:
        return {"depends_on": [], "required_by": []}
        """
        为目标问题生成依赖关系
        
        Args:
            all_questions: 所有问题的列表
            qid: 目标问题的ID
            
        Returns:
            dict: 包含depends_on和required_by的依赖关系字典
        """
        
        # 获取prompt模板
        prompt_get_dependency = self.prompt_updater.get("get_dependency", "")
        if not prompt_get_dependency:
            logging.warning("未找到依赖关系生成的提示词模板")
            return {"depends_on": [], "required_by": []}
        
        # 找到目标问题
        target_question = None
        other_questions_data = []
        
        for q in all_questions:
            if q["id"] == qid:
                target_question = q["description"]
            else:
                other_questions_data.append({
                    "id": q["id"],
                    "description": q["description"]
                })
        
        if not target_question:
            logging.error(f"在all_questions中未找到ID为{qid}的问题")
            return {"depends_on": [], "required_by": []}
        
        # 准备其他问题的格式化字符串
        other_questions_str = json.dumps(other_questions_data, ensure_ascii=False, indent=2)
        
        # 填充提示词模板
        prompt = prompt_get_dependency.format(
            target_question=target_question,
            other_questions=other_questions_str
        )
        
        # 发送请求到VLM API
        response = self.vlm.request_with_retry(image=None, prompt=prompt)[0]
        
        # 从响应中提取JSON部分
        try:
            # 寻找JSON格式的依赖关系
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                dependency = json.loads(json_str)
                
                # 确保返回格式正确
                if not isinstance(dependency.get("depends_on", None), list) or not isinstance(dependency.get("required_by", None), list):
                    logging.error(f"依赖关系格式错误: {dependency}")
                    return {"depends_on": [], "required_by": []}
                    
                return dependency
            else:
                logging.error(f"无法从响应中提取JSON: {response}")
                return {"depends_on": [], "required_by": []}
        except Exception as e:
            logging.error(f"解析依赖关系时出错: {e}, 响应: {response}")
            return {"depends_on": [], "required_by": []}
    
    
    def _merge_dependencies(self, all_questions, new_dependency, qid):
        """
        将新的依赖关系合并到问题列表中
        
        Args:
            all_questions: 所有问题的列表
            new_dependency: 新的依赖关系字典，包含depends_on和required_by
            qid: 目标问题的ID
            
        Returns:
            list: 更新后的问题列表
        """
        # 验证新依赖格式
        if not isinstance(new_dependency, dict) or "depends_on" not in new_dependency or "required_by" not in new_dependency:
            logging.error(f"新依赖格式错误: {new_dependency}")
            return all_questions
        
        # 为目标问题设置其依赖的问题
        target_question = None
        for i, q in enumerate(all_questions):
            if q["id"] == qid:
                # 设置目标问题依赖的问题
                all_questions[i]["dependency"] = new_dependency.get("depends_on", [])
                target_question = q
                break
        
        if not target_question:
            logging.error(f"在all_questions中未找到ID为{qid}的问题")
            return all_questions
        
        # 更新其他问题的依赖关系（将目标问题添加到需要它的问题的依赖列表中）
        for required_by_id in new_dependency.get("required_by", []):
            for i, q in enumerate(all_questions):
                if q["id"] == required_by_id:
                    # 如果依赖列表不存在，创建一个新的
                    if "dependency" not in all_questions[i] or not isinstance(all_questions[i]["dependency"], list):
                        all_questions[i]["dependency"] = []
                    
                    # 添加目标问题ID到依赖列表（如果尚未存在）
                    if qid not in all_questions[i]["dependency"]:
                        all_questions[i]["dependency"].append(qid)
                    break
        
        return all_questions
