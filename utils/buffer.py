# utils/buffer.py

class Buffer:
    def __init__(self):
        self.buffer = []
        self.dag = {}
    
    
    def add_question(self, question):
        required_keys = {"id", "description", "urgency", "scope_type", "status", 
                        "cost_estimate", "reward_estimate", "dependency", "answer", "max_steps", "used_steps", "time"}
        
        if not isinstance(question, dict):
            raise ValueError("Question must be a dictionary.")
            
        if set(question.keys()) != required_keys:
            raise ValueError("Question must contain exactly the required keys: " + ", ".join(required_keys))
        
        # 确保time字段是字典结构
        if not isinstance(question["time"], dict):
            raise ValueError("The 'time' field must be a dictionary.")
            
        self.buffer.append(question)
    
    
    def set_status(self, question_id, status):
        if status not in {"pending", "ready", "completed", "answered", "in_progress"}:
            raise ValueError("Invalid status. Must be one of: 'pending', 'ready', 'completed', 'answered', 'in_progress'.")
        for question in self.buffer:
            if question["id"] == question_id:
                question["status"] = status
                return
        raise ValueError(f"Question with id {question_id} not found in buffer.")
    
    
    def set_answer(self, question_id, answer):
        for question in self.buffer:
            if question["id"] == question_id:
                question["answer"] = answer
                return
        raise ValueError(f"Question with id {question_id} not found in buffer.")
    
    
    def get_question_by_id(self, question_id):
        for question in self.buffer:
            if question["id"] == question_id:
                return question
        raise ValueError(f"Question with id {question_id} not found in buffer.")
    
    
    def get_pending_and_ready_questions(self):
        return [q for q in self.buffer if q["status"] in {"pending", "ready"}]
    
    
    def write_latest_questions(self, new_questions):
        required_keys = {"id", "description", "urgency", "scope_type", "status", 
                        "cost_estimate", "reward_estimate", "dependency", "answer", "max_steps", "used_steps", "time"}
        
        if not isinstance(new_questions, list):
            raise ValueError("new_questions must be a list.")
        
        for question in new_questions:
            if not isinstance(question, dict):
                raise ValueError("Each question must be a dictionary.")
                
            if set(question.keys()) != required_keys:
                raise ValueError("Each question must contain exactly the required keys: " + ", ".join(required_keys))
            
            # 确保time字段是字典结构
            if not isinstance(question["time"], dict):
                raise ValueError("The 'time' field must be a dictionary.")
        
        for new_question in new_questions:
            question_id = new_question["id"]
            found = False
            
            # 查找是否已存在相同id的问题
            for i, existing_question in enumerate(self.buffer):
                if existing_question["id"] == question_id:
                    # 如果找到，更新现有问题
                    self.buffer[i] = new_question
                    found = True
                    break
            
            # 如果没有找到，直接添加到buffer
            if not found:
                self.buffer.append(new_question)
    
    
    def build_dag(self):
        # 筛选等待的问题
        waiting_questions = [
            q for q in self.buffer if q.get("status") in {"pending", "ready"}
        ]
        waiting_qids = {q["id"] for q in waiting_questions}

        # 使用邻接表，表达“问题A依赖于哪些问题”和“哪些问题依赖于问题A”
        self.dag = {
            "depends_on": {},  # question_id -> set of question_ids it depends on
            "required_by": {}  # question_id -> set of question_ids that depend on it
        }

        # 初始化邻接表，只为符合条件的问题创建节点
        for question in waiting_questions:
            qid = question["id"]
            
            original_deps = set(question.get("dependency", []))
            valid_deps = original_deps.intersection(waiting_qids)
            
            self.dag["depends_on"][qid] = valid_deps
            self.dag["required_by"].setdefault(qid, set())

        # 构建 required_by 关系（反向边）
        for question in waiting_questions:
            qid = question["id"]
            for dep_id in self.dag["depends_on"][qid]:
                self.dag["required_by"][dep_id].add(qid)
                
        return self.dag
    
    
    def calculate_status(self):
        """
        基于DAG中的依赖关系，计算每个问题的状态（"pending"或"ready"）
        """
        # 确保DAG已经构建
        if not self.dag:
            self.build_dag()
        
        # 遍历所有等待中的问题
        waiting_questions = [
            q for q in self.buffer if q.get("status") in {"pending", "ready"}
        ]
        
        for question in waiting_questions:
            qid = question["id"]
            
            # 如果该问题在DAG中，检查其依赖
            if qid in self.dag["depends_on"]:
                # 如果没有依赖，状态设为"ready"；否则设为"pending"
                if not self.dag["depends_on"][qid]:  # 空集合表示没有依赖
                    question["status"] = "ready"
                else:
                    question["status"] = "pending"


    def get_size(self):
        return len(self.buffer)


    def clear(self):
        self.buffer = []
        self.dag = {}
    
    
    def get_buffer(self):
        return self.buffer.copy()
    
    
    def get_dag(self):
        return self.dag.copy()
