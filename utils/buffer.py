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
        
        # Ensure the time field is a dictionary structure
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
            
            # Ensure the time field is a dictionary structure
            if not isinstance(question["time"], dict):
                raise ValueError("The 'time' field must be a dictionary.")
        
        for new_question in new_questions:
            question_id = new_question["id"]
            found = False
            
            # Check if a question with the same id already exists
            for i, existing_question in enumerate(self.buffer):
                if existing_question["id"] == question_id:
                    # If found, update the existing question
                    self.buffer[i] = new_question
                    found = True
                    break
            
            # If not found, add directly to buffer
            if not found:
                self.buffer.append(new_question)
    
    
    def build_dag(self):
        # Filter waiting questions
        waiting_questions = [
            q for q in self.buffer if q.get("status") in {"pending", "ready"}
        ]
        waiting_qids = {q["id"] for q in waiting_questions}

        # Use adjacency list to express "which questions question A depends on" and "which questions depend on question A"
        self.dag = {
            "depends_on": {},  # question_id -> set of question_ids it depends on
            "required_by": {}  # question_id -> set of question_ids that depend on it
        }

        # Initialize adjacency list, create nodes only for qualified questions
        for question in waiting_questions:
            qid = question["id"]
            
            original_deps = set(question.get("dependency", []))
            valid_deps = original_deps.intersection(waiting_qids)
            
            self.dag["depends_on"][qid] = valid_deps
            self.dag["required_by"].setdefault(qid, set())

        # Build required_by relationships (reverse edges)
        for question in waiting_questions:
            qid = question["id"]
            for dep_id in self.dag["depends_on"][qid]:
                self.dag["required_by"][dep_id].add(qid)
                
        return self.dag
    
    
    def calculate_status(self):
        """
        Calculate the status of each question ("pending" or "ready") based on dependency relationships in the DAG
        """
        # Ensure DAG has been built
        if not self.dag:
            self.build_dag()
        
        # Iterate through all waiting questions
        waiting_questions = [
            q for q in self.buffer if q.get("status") in {"pending", "ready"}
        ]
        
        for question in waiting_questions:
            qid = question["id"]
            
            # If this question is in the DAG, check its dependencies
            if qid in self.dag["depends_on"]:
                # If no dependencies, set status to "ready"; otherwise set to "pending"
                if not self.dag["depends_on"][qid]:  # Empty set means no dependencies
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
