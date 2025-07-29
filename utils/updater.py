# utils/updater.py

import logging

from utils.buffer import Buffer
from vlm_openai import VLM_OpenAI


class Updater:
    def __init__(self, config):
        updater_config = config.get("updater", {})
        self.enable_cost_estimate = updater_config.get("enable_cost_estimate", False)
        self.enable_reward_estimate = updater_config.get("enable_reward_estimate", False)
        
        # Set up VLM
        self.prompt_updater = config.get("prompt", {}).get("updater", {})
        model_openai = config.get("vlm", {}).get("model_openai", "gpt-4.1")
        self.vlm = VLM_OpenAI(model_name=model_openai)
        # Use: response = self.vlm.request_with_retry(image=None, prompt=prompt)[0]
        
        # Set up Buffer
        self.buffer = Buffer()


    def add_question(self, question):
        # Finishing Module调用该方法加入一个问题
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
        
        self.buffer.write_latest_questions(all_questions)
        
        self.buffer.build_dag()
        self.buffer.calculate_status()
    
    
    def complete_question(self, question):
        # Stopping Module调用该方法表示已经完成一个问题
        # 标记为completed
        question_in_progress = self.buffer.get_question_by_id(question["id"])
        if question_in_progress["status"] != "in_progress":
            raise ValueError(f"Question {question["id"]} is not in progress. Cannot possibly be completed.")
        self.buffer.set_status(question_in_progress["id"], "completed")
        
        # TODO: 当一个问题的状态由ready变为in_progress时，执行build_dag()和calculate_status()
        # self.buffer.build_dag()
        # self.buffer.calculate_status()
    
    
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
