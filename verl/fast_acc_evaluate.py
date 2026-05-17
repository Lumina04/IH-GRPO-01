import re
import torch
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
import json
import os
import datasets
from datasets import load_from_disk, load_dataset, Dataset
from utils.reward_score.math import compute_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def compute_avg_acc(completion, answer):
    acc_li = []
    for response in completion:
        acc = compute_score(response, answer)
        acc_li.append(acc)
    avg_acc = sum(acc_li) / len(acc_li)
    return avg_acc

def get_answer(doc, data_source):
    if data_source in ["mmlu_pro", "date_understanding", "formal_fallacies", "logical_deduction_seven_objects"]:
        return str(doc["answer"])
    elif data_source == "logiqa":
        answer_li = ['A', 'B', 'C', 'D']
        return str(answer_li[doc['gold'][0]])
    else:
        raise Exception("no such data...")

def doc_to_text(doc, data_source):
    if data_source in ["mmlu_pro"]:
        return doc['input'] + " Let's think step by step and Only output the final choice within \\boxed{}. /no_think"
    elif data_source == "logiqa":
        return doc['query'][:-15] + ". Let's think step by step and Only output the final choice within \\boxed{}. /no_think"
    elif data_source in ["date_understanding", "formal_fallacies", "logical_deduction_seven_objects"]:
        return doc['input_question'] + " Let's think step by step and Only output the final choice within \\boxed{}. /no_think"
    else:
        raise Exception("no such data...")

def generate_sample(tokenizer, input_txt, sampling_params):
    prompt = [
        {"role": "user", "content": input_txt},
    ]
    prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    outputs = llm.generate(prompt, sampling_params)
    return_text = []
    for i in range(len(outputs[0].outputs)):
        output_text = outputs[0].outputs[i].text
        return_text.append(output_text)
    return return_text

def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip().replace(",", "")
        try:
            return eval(match_str)
        except:
            return INVALID_ANS
    else:
        return INVALID_ANS

def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS

def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Evaluate HF-format checkpoint on data")
    parser.add_argument("--data-source", type=str, default="logiqa",
                    choices=["logiqa", "mmlu_pro", "date_understanding", "formal_fallacies", "logical_deduction_seven_objects"],
                    help="Which dataset to evaluate on")
    parser.add_argument(
        "--output-name",
        type=str,
        default="qwen3_1.7b_base.jsonl",
        help="Output jsonl filename (only name, not full path)"
    )
    parser.add_argument("-c", "--checkpoint-path", type=str,
                        default="xxx",
                        help="Path to the model checkpoint")

    args = parser.parse_args()
    
    base_data_path = "xxx"
    test_data_path_map = {
        "mmlu_pro": base_data_path+"mmlu-pro.parquet",
        "logiqa":base_data_path+"logiqa.parquet",
        "date_understanding": base_data_path+"date_understanding.parquet",
        "formal_fallacies": base_data_path+"formal_fallacies.parquet",
        "logical_deduction_seven_objects": base_data_path+"logical_deduction_seven_objects.parquet",
    }

    data_source = args.data_source
    output_dir = f"xxx"
    checkpoint_lower = args.checkpoint_path.lower()
    if "1.7b" in checkpoint_lower:
        output_dir += f"qwen3_1.7b/{data_source}/"
    elif "4b" in checkpoint_lower:
        output_dir += f"qwen3_4b/{data_source}/"
    elif "8b" in checkpoint_lower:
        output_dir += f"qwen3_8b/{data_source}/"
    else:
        raise Exception("no such model")
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{args.output_name}"
    
    test_data_path = test_data_path_map[data_source]
    if test_data_path:
        if test_data_path.endswith(".parquet"):
            dataset = datasets.Dataset.from_parquet(test_data_path)
        elif test_data_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=test_data_path, split="train")
        elif test_data_path.endswith(".json"):
            with open(test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        else:
            dataset = load_from_disk(test_data_path)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset(data_source, "main", download_config=config)["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True, local_files_only=True)

    llm = LLM(
        model=args.checkpoint_path,
        tokenizer=args.checkpoint_path, # Explicitly pass tokenizer path
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=None,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.7,
    )
    sampling_params = SamplingParams(
        n=1,          # Number of output sequences to return for each prompt
        temperature=1.0,
        max_tokens=8000,
        top_k=-1,
        top_p=0.7
    )

    acc_res = []
    print("Running evaluation...")
    for i, doc in enumerate(tqdm(dataset, desc="Evaluating", dynamic_ncols=True)):
        try:
            context = doc_to_text(doc, data_source)
            answer = get_answer(doc, data_source)
            completion = generate_sample(tokenizer, context, sampling_params)
            avg_acc = compute_avg_acc(completion, answer)
        except Exception as e:
            print(f"[ERROR] Sample {i} failed: {e}")
            avg_acc = 0

        acc_res.append(avg_acc)
        if (i + 1) % 10 == 0:
            current_acc = float(np.mean(acc_res))
            print(f"[{i + 1}/{len(dataset)}] Current Accuracy: {current_acc:.4f}")

    final_accuracy = float(np.mean(acc_res))
    with jsonlines.open(output_file, mode="w") as writer:
        writer.write({
            "data_source": data_source,
            "num_samples": len(acc_res),
            "final_accuracy": final_accuracy
        })

    print(f"\nFinal Accuracy: {final_accuracy:.4f}")
    print(f"Saved result to: {output_file}")
