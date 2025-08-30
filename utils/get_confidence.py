# utils/get_confidence.py

import logging
from utils.vlm_api import VLM_API
from utils.image_processor import decode_image


def get_confidence(question_desc, image, kb, prompt_get_confidence, model_name="qwen/qwen2.5-vl-72b-instruct", server="openrouter", base_url=None, api_key=None):
    """
    Call the large model to calculate the confidence of whether the question can be answered
    
    Args:
        question_desc (str): Question description
        image (str): Image data, base64 encoded string, can be empty
        kb (list): List of items retrieved from memory, can be empty
        prompt_get_confidence (str): Prompt template
        model_name (str): Model name to use
        server (str): Server type
        base_url (str): Base URL
        api_key (str): API key
        
    Returns:
        float: Confidence, range [0,1]
    """    
    # Instantiate VLM
    vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
    
    # Construct prompt
    prompt = prompt_get_confidence.format(question_desc)
    
    # Mapping table of choices and values
    choices_mapping = {
        "A": 0.1,
        "B": 0.3,
        "C": 0.5,
        "D": 0.7,
        "E": 0.9
    }
    
    # Maximum 5 retries
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            # Call VLM
            response = vlm.request_with_retry(image=image, prompt=prompt, kb=kb)[0]
            
            # Parse response to get confidence
            confidence = choices_mapping.get(response.strip().upper(), -1.0)
            
            if confidence != -1.0:
                # Successfully parsed, ensure confidence is in [0,1] range and return
                confidence = max(0.0, min(1.0, confidence))
                if attempt > 0:
                    logging.info(f"Attempt {attempt + 1} successfully parsed confidence: {confidence}")
                return confidence
            
            else:
                logging.warning(f"Attempt {attempt + 1} failed to parse confidence, VLM response: {response}")
        
        except ValueError as e:
            logging.warning(f"Error occurred while parsing confidence on attempt {attempt + 1}: {e}, VLM response: {response}")
        
        except Exception as e:
            logging.error(f"Error occurred while calling VLM on attempt {attempt + 1}: {e}")
    
    # All retries failed, return default value
    logging.error(f"After {max_retries} attempts, still unable to parse confidence from VLM response, returning default value 0.0")
    return 0.0


def get_tryout_confidence(question_desc, image, kb, prompt_get_tryout_answer, prompt_get_tryout_confidence, model_name="qwen/qwen2.5-vl-72b-instruct", server="openrouter", base_url=None, api_key=None):
    """
    First let the large model try to answer the question, then judge confidence based on the guessed answer
    
    Args:
        question_desc (str): Question description
        image (str): Image data, base64 encoded string, can be empty
        kb (list): List of items retrieved from memory, can be empty
        prompt_get_tryout_answer (str): Prompt template
        prompt_get_tryout_confidence (str): Prompt template
        model_name (str): Model name to use
        server (str): Server type
        base_url (str): Base URL
        api_key (str): API key
        
    Returns:
        float: Confidence, range [0,1]
    """
    # Instantiate VLM
    vlm = VLM_API(model_name=model_name, server=server, base_url=base_url, api_key=api_key)
    
    # Construct first prompt - get guessed answer
    prompt_answer = prompt_get_tryout_answer.format(question_desc)
    
    # Mapping table of choices and values
    choices_mapping = {
        "A": 0.1,
        "B": 0.3,
        "C": 0.5,
        "D": 0.7,
        "E": 0.9
    }
    
    # Maximum 5 retries
    max_retries = 5
    
    # First step: get guessed answer
    answer_tryout = None
    
    for attempt in range(max_retries):
        try:
            # Call VLM to get guessed answer
            response = vlm.request_with_retry(image=image, prompt=prompt_answer, kb=kb)[0]
            
            if response is None:
                logging.warning(f"Attempt {attempt + 1} to get guessed answer, VLM response is None")
                continue
                
            # Parse response to get guessed answer
            answer_tryout = response.strip()
            
            if answer_tryout:
                logging.info(f"Successfully obtained guessed answer: {answer_tryout}")
                if attempt > 0:
                    logging.info(f"Attempt {attempt + 1} successfully obtained guessed answer")
                break
            
            else:
                logging.warning(f"Attempt {attempt + 1} failed to get guessed answer, VLM response is empty string")
        
        except Exception as e:
            logging.error(f"Error occurred while calling VLM to get guessed answer on attempt {attempt + 1}: {e}")
    
    # If unable to get guessed answer, return default confidence
    if not answer_tryout:
        logging.error(f"After {max_retries} attempts, still unable to get guessed answer, returning default value 0.0")
        return 0.0
    
    # Second step: get confidence based on guessed answer
    # Construct second prompt - get confidence
    prompt_confidence = prompt_get_tryout_confidence.format(question_desc=question_desc, tryout_answer=answer_tryout)
    
    for attempt in range(max_retries):
        try:
            # Call VLM to get confidence
            response = vlm.request_with_retry(image=image, prompt=prompt_confidence, kb=kb)[0]
            
            if response is None:
                logging.warning(f"Attempt {attempt + 1} to get confidence, VLM response is None")
                continue
                
            # Parse response to get confidence
            confidence = choices_mapping.get(response.strip().upper(), -1.0)
            
            if confidence != -1.0:
                # Successfully parsed, ensure confidence is in [0,1] range and return
                confidence = max(0.0, min(1.0, confidence))
                if attempt > 0:
                    logging.info(f"Attempt {attempt + 1} successfully parsed confidence: {confidence}")
                return confidence
            
            else:
                logging.warning(f"Attempt {attempt + 1} failed to parse confidence, VLM response: {response}")
        
        except ValueError as e:
            logging.warning(f"Error occurred while parsing confidence on attempt {attempt + 1}: {e}, VLM response: {response}")
        
        except Exception as e:
            logging.error(f"Error occurred while calling VLM to get confidence on attempt {attempt + 1}: {e}")
    
    # All retries failed, return default value
    logging.error(f"After {max_retries} attempts, still unable to parse confidence from VLM response, returning default value 0.0")
    return 0.0



"""
kb format:
[
    {
        "id": str,
        "text": str,
        "image": str,  # base64 encoded image string
    },
    ...
]
"""
