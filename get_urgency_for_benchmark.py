# get_urgency_for_benchmark.py

import os
import glob
import yaml
import json
import re
import time
import base64
import httpx
import openai
from io import BytesIO
from PIL import Image


class VLM_API:
    """Simplified VLM API calling class, only supports pure text requests"""
    
    def __init__(self, model_name="gpt-oss:20b", server="ollama", base_url="http://100.88.238.80:11434/v1", api_key=None):
        self.model_name = model_name
        self.server = server
        self.base_url = base_url
        self.api_key = api_key
        
        # Initialize proxy settings
        proxies = {
            "http://": "socks5://127.0.0.1:7897",
            "https://": "socks5://127.0.0.1:7897"
        } if os.environ.get('ALL_PROXY') else None
        
        # Initialize API client
        if self.server == "openai":
            self.api_key = os.environ.get('OPENAI_API_KEY') if not api_key else api_key
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.openai.com/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
        elif self.server == "openrouter":
            self.api_key = os.environ.get('OPENROUTER_API_KEY') if not api_key else api_key
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
        elif self.server == "dashscope":
            self.api_key = os.environ.get('DASHSCOPE_API_KEY') if not api_key else api_key
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
        elif self.server == "ollama":
            self.client = openai.OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1" if not self.base_url else self.base_url,
                http_client=httpx.Client(proxies=proxies) if proxies else None
            )
    
    def request_with_retry(self, prompt, retries=3):
        """Send pure text request with retry support"""
        def exponential_backoff(attempt):
            return min(2 ** attempt, 60)
        
        for attempt in range(retries):
            try:
                return self.requests_api_only_text(prompt)
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = exponential_backoff(attempt)
                    print(f"Request failed, retrying in {wait_time} seconds... Error: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
    
    def requests_api_only_text(self, prompt):
        """Pure text API request"""
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Use OpenAI official client to send request
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=1000,
            temperature=0.0
        )
        
        # Extract response content
        return [response.choices[0].message.content]


def get_urgency(desc, model_name="gpt-oss:20b", server="ollama", base_url="http://100.88.238.80:11434/v1", api_key=None):
    """
    Parse problem description and extract urgency value using large model
    
    Args:
        desc (str): Natural language description of the problem
        model_name (str): Model name to use
        server (str): Server type
        base_url (str): Base URL
        api_key (str): API key
        
    Returns:
        float: urgency value, range [0,1]
    """
    # Prompt template
    prompt_template = """
    You are an AI assistant tasked with parsing natural language questions into structured data for an Embodied Question Answering (EQA) system in a multi-story residential environment. Your goal is to extract key information from a question and represent it in a JSON object containing an `urgency` field.

    - **urgency**: A float value between 0 and 1 indicating the urgency of the question. Safety-related questions have higher urgency (e.g., 0.9), functionality-related questions have medium urgency (e.g., 0.5), and general information questions have lower urgency (e.g., 0.2).

    Here are some examples to guide you:

    <Example 1>
    - **Question**: "What color is the sofa in the living room?"
    - **Structured Output**:
    ```json
    {
        "urgency": 0.2
    }
    ```

    <Example 2>
    - **Question**: "Is the TV in the living room turned on?"
    - **Structured Output**:
    ```json
    {
        "urgency": 0.4
    }
    ```

    <Example 3>
    - **Question**: "Where can you find my cell phone?"
    - **Structured Output**:
    ```json
    {
        "urgency": 0.6
    }
    ```

    <Example 4>
    - **Question**: "Is there any fire risk in my house?"
    - **Structured Output**:
    ```json
    {
        "urgency": 0.9
    }
    ```

    Now, please parse the following question: {original_question}

    Please directly provide the structured output in JSON format and DO NOT include any additional text or explanations.
    """
    
    max_retries = 3
    
    # Replace question in template
    prompt = prompt_template.replace("{original_question}", desc)
    
    # Instantiate VLM API
    vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            # Call large model to get response
            response = vlm.request_with_retry(prompt=prompt)[0]
            
            # Extract JSON part
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response.strip()
            
            # Parse JSON
            parsed_data = json.loads(json_str)
            
            # Extract urgency
            urgency = parsed_data.get('urgency')
            
            # Validate value range
            if not isinstance(urgency, (int, float)) or urgency < 0 or urgency > 1:
                print(f"Attempt {attempt + 1}: urgency value does not meet rules (should be a numeric value in [0,1] range): {urgency}")
                continue
            
            # Return valid urgency value
            return float(urgency)
            
        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1} failed, returned invalid JSON: {response}")
        except Exception as e:
            print(f"Error occurred in attempt {attempt + 1}: {e}")
    
    # If all attempts fail, return default value
    print(f"After {max_retries} attempts, unable to get valid urgency value, using default value 0.5")
    return 0.5


def process_yaml_files(benchmark_dir="data/benchmark", model_config=None, parse_from_scratch=False):
    """Process all YAML files"""
    if model_config is None:
        model_config = {
            "model_name": "gpt-oss:20b",
            "server": "ollama", 
            "base_url": "http://100.88.238.80:11434/v1",
            "api_key": None
        }
    
    # Get all YAML files and sort by filename
    yaml_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.yaml")))
    
    # Select files to process
    yaml_files = yaml_files[40:]
    
    if not yaml_files:
        print(f"Warning: No YAML files found in {benchmark_dir} directory")
        return
    
    print(f"Found {len(yaml_files)} YAML files, starting to process in filename order...")
    
    # Process each file
    for file_path in yaml_files:
        print(f"\nProcessing file: {file_path}")
        
        try:
            # Read YAML file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Process questions_init
            if 'questions_init' in data and data['questions_init']:
                for i, question in enumerate(data['questions_init']):
                    # Check if valid urgency field already exists
                    has_valid_urgency = False
                    if 'urgency' in question:
                        if isinstance(question['urgency'], (int, float)) and 0 <= question['urgency'] <= 1:
                            print(f"Init question {i+1} already has valid urgency value: {question['urgency']}")
                            has_valid_urgency = True
                        else:
                            print(f"Init question {i+1} has invalid urgency value: {question['urgency']}, will recalculate")
                    
                    if parse_from_scratch or not has_valid_urgency:
                        # Extract question text
                        question_text = question.get('question', '')
                        if not question_text:
                            print(f"Warning: Init question {i+1} has no question field or is empty")
                            question['urgency'] = 0.5
                            continue
                        
                        print(f"Getting urgency for Init question {i+1}: {question_text[:50]}...")
                        # Get urgency value
                        urgency = get_urgency(
                            question_text,
                            model_name=model_config["model_name"],
                            server=model_config["server"],
                            base_url=model_config["base_url"],
                            api_key=model_config["api_key"]
                        )
                        
                        # Update question object
                        question['urgency'] = urgency
                        print(f"Set urgency value for Init question {i+1} to: {urgency}")
            
            # Process questions_follow_up
            if 'questions_follow_up' in data and data['questions_follow_up']:
                for i, question in enumerate(data['questions_follow_up']):
                    # Check if valid urgency field already exists
                    has_valid_urgency = False
                    if 'urgency' in question:
                        if isinstance(question['urgency'], (int, float)) and 0 <= question['urgency'] <= 1:
                            print(f"Follow-up question {i+1} already has valid urgency value: {question['urgency']}")
                            has_valid_urgency = True
                        else:
                            print(f"Follow-up question {i+1} has invalid urgency value: {question['urgency']}, will recalculate")
                    
                    if parse_from_scratch or not has_valid_urgency:
                        # Extract question text
                        question_text = question.get('question', '')
                        if not question_text:
                            print(f"Warning: Follow-up question {i+1} has no question field or is empty")
                            question['urgency'] = 0.5
                            continue
                        
                        print(f"Getting urgency for Follow-up question {i+1}: {question_text[:50]}...")
                        # Get urgency value
                        urgency = get_urgency(
                            question_text,
                            model_name=model_config["model_name"],
                            server=model_config["server"],
                            base_url=model_config["base_url"],
                            api_key=model_config["api_key"]
                        )
                        
                        # Update question object
                        question['urgency'] = urgency
                        print(f"Set urgency value for Follow-up question {i+1} to: {urgency}")
            
            # Save updated YAML file
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"Successfully updated file: {file_path}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


def main():
    """Main function"""
    # You can modify model configuration here
    model_config = {
        "model_name": "gpt-oss:20b",  # Model name
        "server": "ollama",           # Server type: openai, openrouter, dashscope, ollama
        "base_url": "http://100.88.238.80:11434/v1",  # Base URL
        "api_key": None               # API key
    }
    
    benchmark_dir = "data/benchmark"  # Benchmark folder path
    
    parse_from_scratch = False  # Whether to parse all files from scratch
    
    # Start processing files
    process_yaml_files(benchmark_dir, model_config, parse_from_scratch)
    print("\nProcessing completed!")


if __name__ == "__main__":
    main()
