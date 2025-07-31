# utils/updater.py

import logging
import math
import json

from utils.buffer import Buffer
from vlm_openai import VLM_OpenAI


class Updater:
    def __init__(self, config):
        updater_config = config.get("updater", {})
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
        model_openai = config.get("vlm", {}).get("model_openai", "gpt-4.1")
        self.vlm = VLM_OpenAI(model_name=model_openai)
        # Use: response = self.vlm.request_with_retry(image=None, prompt=prompt)[0]
        
        # Set up Buffer
        self.buffer = Buffer()
        
        # 存储当前优先级最高的问题
        self.highest_priority_question = None
        self.highest_priority_score = float('-inf')


    def add_question(self, question):
        # Finishing Module调用该方法加入一个问题
        self.highest_priority_question = None
        self.highest_priority_score = float('-inf')
        
        # 先逐一计算cost和reward，然后一次性计算dependency，最后status由buffer中的calculate_status()方法生成
        original_questions = self.buffer.get_pending_and_ready_questions()
        
        all_questions = original_questions.copy()
        all_questions.append(question)
        for q in all_questions:
            q["cost_estimate"] = self._get_cost_estimate(q) if self.enable_cost_estimate else 0.0
            q["reward_estimate"] = self._get_reward_estimate(q) if self.enable_reward_estimate else 0.0
        
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
    
    
    def get_highest_priority_question(self):
        """获取当前优先级最高的问题"""
        if self.highest_priority_question is None:
            logging.info("没有待处理或就绪的问题。")
            return None
        
        # 如果最高优先级问题的状态不是就绪，则重新计算优先级分数
        if self.highest_priority_question["status"] != "ready":
            logging.info(f"最高优先级问题ID: {self.highest_priority_question['id']} 状态为 {self.highest_priority_question['status']}，重新计算优先级分数。")
            self._update_priority_scores()
        
        return self.highest_priority_question
    
    
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
    
    
    def _get_cost_estimate(self, question):
        # TODO: 具体的cost计算方式
        prompt_get_cost_estimate = self.prompt_updater.get("get_cost_estimate", "")
        original_questions = self.buffer.get_pending_and_ready_questions()
        # 伪函数
        prompt = concatinate(prompt_get_cost_estimate, question, original_questions)
        
        response = self.vlm.request_with_retry(image=None, prompt=prompt)[0]
        try:
            cost_estimate = float(response)
        except ValueError:
            logging.info(f"Error parsing cost estimate from response: {response}")
            cost_estimate = 0.0
        return cost_estimate
    
    
    def _get_reward_estimate(self, question):
        # TODO: 具体的reward计算方式
        prompt_get_reward_estimate = self.prompt_updater.get("get_reward_estimate", "")
        original_questions = self.buffer.get_pending_and_ready_questions()
        # 伪函数
        prompt = concatinate(prompt_get_reward_estimate, question, original_questions)
        
        response = self.vlm.request_with_retry(image=None, prompt=prompt)[0]
        try:
            reward_estimate = float(response)
        except ValueError:
            logging.info(f"Error parsing reward estimate from response: {response}")
            reward_estimate = 0.0
        return reward_estimate
    
    
    def _get_new_dependency(self, all_questions, qid):
        pass
    
    
    def _merge_dependencies(self, all_questions, new_dependency, qid):
        pass
