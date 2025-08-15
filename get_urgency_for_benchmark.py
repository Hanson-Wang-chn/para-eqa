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
    """简化版的VLM API调用类，仅支持纯文本请求"""
    
    def __init__(self, model_name="gpt-oss:20b", server="ollama", base_url="http://100.88.238.80:11434/v1", api_key=None):
        self.model_name = model_name
        self.server = server
        self.base_url = base_url
        self.api_key = api_key
        
        # 初始化代理设置
        proxies = {
            "http://": "socks5://127.0.0.1:7897",
            "https://": "socks5://127.0.0.1:7897"
        } if os.environ.get('ALL_PROXY') else None
        
        # 初始化API客户端
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
        """发送纯文本请求并支持重试"""
        def exponential_backoff(attempt):
            return min(2 ** attempt, 60)
        
        for attempt in range(retries):
            try:
                return self.requests_api_only_text(prompt)
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = exponential_backoff(attempt)
                    print(f"请求失败，{wait_time}秒后重试... 错误: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
    
    def requests_api_only_text(self, prompt):
        """纯文本API请求"""
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
        
        # 使用OpenAI官方客户端发送请求
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=1000,
            temperature=0.0
        )
        
        # 提取响应内容
        return [response.choices[0].message.content]


def get_urgency(desc, model_name="gpt-oss:20b", server="ollama", base_url="http://100.88.238.80:11434/v1", api_key=None):
    """
    解析问题描述，使用大模型提取urgency值
    
    Args:
        desc (str): 问题的自然语言描述
        model_name (str): 使用的模型名称
        server (str): 服务器类型
        base_url (str): 基础URL
        api_key (str): API密钥
        
    Returns:
        float: urgency值，范围[0,1]
    """
    # 提示词模板
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
    
    # 替换模板中的问题
    prompt = prompt_template.replace("{original_question}", desc)
    
    # 实例化VLM API
    vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            # 调用大模型获取回答
            response = vlm.request_with_retry(prompt=prompt)[0]
            
            # 提取JSON部分
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response.strip()
            
            # 解析JSON
            parsed_data = json.loads(json_str)
            
            # 提取urgency
            urgency = parsed_data.get('urgency')
            
            # 验证取值范围
            if not isinstance(urgency, (int, float)) or urgency < 0 or urgency > 1:
                print(f"第{attempt + 1}次尝试: urgency值不符合规则 (应为[0,1]范围内的数值): {urgency}")
                continue
            
            # 返回有效的urgency值
            return float(urgency)
            
        except json.JSONDecodeError:
            print(f"第{attempt + 1}次尝试失败，返回的不是有效JSON: {response}")
        except Exception as e:
            print(f"第{attempt + 1}次尝试发生错误: {e}")
    
    # 如果多次尝试都失败，返回默认值
    print(f"经过{max_retries}次尝试，无法获得有效的urgency值，使用默认值0.5")
    return 0.5


def process_yaml_files(benchmark_dir="data/benchmark", model_config=None, parse_from_scratch=False):
    """处理所有YAML文件"""
    if model_config is None:
        model_config = {
            "model_name": "gpt-oss:20b",
            "server": "ollama", 
            "base_url": "http://100.88.238.80:11434/v1",
            "api_key": None
        }
    
    # 获取所有YAML文件并按文件名排序
    yaml_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.yaml")))
    
    # TODO: 选择要处理的文件
    yaml_files = yaml_files[30:]
    
    if not yaml_files:
        print(f"警告: 在{benchmark_dir}目录中未找到YAML文件")
        return
    
    print(f"找到{len(yaml_files)}个YAML文件，按文件名顺序开始处理...")
    
    # 处理每个文件
    for file_path in yaml_files:
        print(f"\n处理文件: {file_path}")
        
        try:
            # 读取YAML文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 处理questions_init
            if 'questions_init' in data and data['questions_init']:
                for i, question in enumerate(data['questions_init']):
                    # 检查是否已有有效的urgency字段
                    has_valid_urgency = False
                    if 'urgency' in question:
                        if isinstance(question['urgency'], (int, float)) and 0 <= question['urgency'] <= 1:
                            print(f"Init问题{i+1}已有有效urgency值: {question['urgency']}")
                            has_valid_urgency = True
                        else:
                            print(f"Init问题{i+1}的urgency值无效: {question['urgency']}，将重新计算")
                    
                    if parse_from_scratch or not has_valid_urgency:
                        # 提取问题文本
                        question_text = question.get('question', '')
                        if not question_text:
                            print(f"警告: Init问题{i+1}没有question字段或为空")
                            question['urgency'] = 0.5
                            continue
                        
                        print(f"为Init问题{i+1}获取urgency: {question_text[:50]}...")
                        # 获取urgency值
                        urgency = get_urgency(
                            question_text,
                            model_name=model_config["model_name"],
                            server=model_config["server"],
                            base_url=model_config["base_url"],
                            api_key=model_config["api_key"]
                        )
                        
                        # 更新问题对象
                        question['urgency'] = urgency
                        print(f"已设置Init问题{i+1}的urgency值为: {urgency}")
            
            # 处理questions_follow_up
            if 'questions_follow_up' in data and data['questions_follow_up']:
                for i, question in enumerate(data['questions_follow_up']):
                    # 检查是否已有有效的urgency字段
                    has_valid_urgency = False
                    if 'urgency' in question:
                        if isinstance(question['urgency'], (int, float)) and 0 <= question['urgency'] <= 1:
                            print(f"Follow-up问题{i+1}已有有效urgency值: {question['urgency']}")
                            has_valid_urgency = True
                        else:
                            print(f"Follow-up问题{i+1}的urgency值无效: {question['urgency']}，将重新计算")
                    
                    if parse_from_scratch or not has_valid_urgency:
                        # 提取问题文本
                        question_text = question.get('question', '')
                        if not question_text:
                            print(f"警告: Follow-up问题{i+1}没有question字段或为空")
                            question['urgency'] = 0.5
                            continue
                        
                        print(f"为Follow-up问题{i+1}获取urgency: {question_text[:50]}...")
                        # 获取urgency值
                        urgency = get_urgency(
                            question_text,
                            model_name=model_config["model_name"],
                            server=model_config["server"],
                            base_url=model_config["base_url"],
                            api_key=model_config["api_key"]
                        )
                        
                        # 更新问题对象
                        question['urgency'] = urgency
                        print(f"已设置Follow-up问题{i+1}的urgency值为: {urgency}")
            
            # 保存更新后的YAML文件
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            print(f"已成功更新文件: {file_path}")
            
        except Exception as e:
            print(f"处理文件{file_path}时出错: {e}")


def main():
    """主函数"""
    # 你可以在这里修改模型配置
    model_config = {
        "model_name": "gpt-oss:20b",  # 模型名称
        "server": "ollama",           # 服务器类型: openai, openrouter, dashscope, ollama
        "base_url": "http://100.88.238.80:11434/v1",  # 基础URL
        "api_key": None               # API密钥
    }
    
    benchmark_dir = "data/benchmark"  # 基准文件夹路径
    
    parse_from_scratch = False  # 是否从头解析所有文件
    
    # 开始处理文件
    process_yaml_files(benchmark_dir, model_config, parse_from_scratch)
    print("\n处理完成！")


if __name__ == "__main__":
    main()
