import os
import json
import numpy as np
import nltk
import http.client
import json

from bert_score import score
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from tqdm import tqdm

def load_jsons(files_path):
    """
    读取 JSON 文件。

    :param file_path: 文件路径
    :return: JSON 数据
    """
    all_data = []
    for file_path in files_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                all_data.extend(json.load(file))
    return all_data

def calc_bertscore(candidates, references):
    P, R, F1 = score(candidates, references, lang="en", verbose=True)
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

def calc_rougescore(candidates, references, metrics=['rouge1', 'rouge2', 'rougeL']):
    rouge_scores = rouge_scorer.RougeScorer(
        metrics, 
        use_stemmer=True
    )
    scores = []
    results = {metric: {} for metric in metrics}
    for candidate, reference in zip(candidates, references):
        score = rouge_scores.score(candidate, reference)
        scores.append(score)
    
    for metric in metrics:
        results[metric]["precision"] = np.array([score[metric].precision for score in scores]).mean()
        results[metric]["recall"] = np.array([score[metric].recall for score in scores]).mean()
        results[metric]["fmeasure"] = np.array([score[metric].fmeasure for score in scores]).mean()

    return results

def calc_meteorscore(candidates, references):
    scores = []
    for candidate, reference in zip(candidates, references):
        reference_tokens = word_tokenize(reference)
        candidate_tokens = word_tokenize(candidate)
        score = meteor_score([reference_tokens], candidate_tokens)
        scores.append(int(score))
    return np.array(scores).mean()

def calc_gptscore(candidates, references):
    scores = []
    for candidate, reference in tqdm(zip(candidates, references)):
        score = requests_api(candidate, reference)
        try:
            scores.append(int(score))
        except:
            continue
    return np.array(scores).mean()

def requests_api(candidate, reference):
    conn = http.client.HTTPSConnection('api.deerapi.com')
    prompt_text = f"Evaluate the semantic similarity between these two sentences: 1. {candidate}. 2. {reference}.\nAnswering only with a single 0-100 score."
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
            }
            ],
            "max_tokens": 400
        })
    headers = {
        'Authorization': 'sk-lrenmYBYEOQH0rqv9rlMmoTaELkvZni1afswhr6be3tTN44S',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"]

def evaluate(files_path):
    samples = load_jsons(files_path)

    norm_steps = []
    candidates = []
    references = []
    for data in samples:
        meta_data = data['meta']
        step_data = data['step']
        cur_step = step_data[-1]['step']
        max_step = meta_data['max_steps']

        norm_steps.append(cur_step / max_step)
        references.append(meta_data["answer"])
        candidates.append(step_data[-1]["smx_vlm_pred"])

    bert_score = calc_bertscore(candidates, references)
    meteor_score = calc_meteorscore(candidates, references)
    rouge_score = calc_rougescore(candidates, references)
    gpt_score = calc_gptscore(candidates, references)

    print("BERT Score:", bert_score)
    print("METEOR Score:", meteor_score)
    print("ROUGE Score:", rouge_score)
    print("GPT Score:", gpt_score)

    print(f"Average normalized steps: {np.array(norm_steps).mean():.3f}")


if __name__ == '__main__':
    files_path = [
        'results/diff-mt/Qwen2-VL-7B-Instruct-open/Qwen2-VL-7B-Instruct-open_gpu0/results.json',
        'results/diff-mt/Qwen2-VL-7B-Instruct-open/Qwen2-VL-7B-Instruct-open_gpu1/results.json',
        'results/diff-mt/Qwen2-VL-7B-Instruct-open/Qwen2-VL-7B-Instruct-open_gpu2/results.json',
        'results/diff-mt/Qwen2-VL-7B-Instruct-open/Qwen2-VL-7B-Instruct-open_gpu3/results.json',
        'results/diff-mt/Qwen2-VL-7B-Instruct-open/Qwen2-VL-7B-Instruct-open_gpu4/results.json',
        'results/diff-mt/Qwen2-VL-7B-Instruct-open/Qwen2-VL-7B-Instruct-open_gpu5/results.json',
        'results/diff-mt/Qwen2-VL-7B-Instruct-open/Qwen2-VL-7B-Instruct-open_gpu6/results.json',
    ]
    evaluate(files_path)
