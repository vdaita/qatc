from fire import Fire
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from llmlingua import PromptCompressor
from cpc.prompt_compressor import PromptCompressorCPC, ModelType, SamplePreprocessor
import time
import tiktoken
from typing import List
from datasets import load_dataset
from tqdm import tqdm
import json

load_dotenv()

PROMPT = """
Please read the following text and answer the question below.

<text>
{document}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {c_a}
(B) {c_b}
(C) {c_c}
(D) {c_d}

Format your response as follows: "The correct answer is (insert answer here)".
"""

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
tokenizer = tiktoken.get_encoding("cl100k_base")
llm_lingua = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True)
cpc_compressor = PromptCompressorCPC(
    model_type=ModelType.LLAMA
)

def split_string(text: str, num_tokens: 2000) -> List[str]:
    tokenized = tokenizer.encode(text)
    token_chunks = [tokenized[i:max(i + num_tokens, len(tokenized))] for i in range(0, len(tokenized), num_tokens)]
    string_chunks = [tokenizer.decode(token_chunk) for token_chunk in token_chunks]
    return string_chunks

def extract_response(response: str) -> str:
    response = response.replace("*", "")
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_model_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful and intelligent assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

def baseline_llmlingua(context: str, question: str, target_token: int) -> str:
    compressed_prompt = llm_lingua.compress_prompt(
        context_list=[context],
        target_token=target_token,
        force_tokens=["\nPassage:", ".", "?", "\n"]
    )
    return compressed_prompt

def baseline_cpc(context: str, question: str, target_token: int) -> str:
    compressed_prompt = cpc_compressor.compress(
        context=context,
        question=question,
        compression_target_tokens=target_token
    )
    return compressed_prompt

def cpc_sequential(context: str, question: str, target_token: int) -> str:
    chunks = split_string(context, 6000)
    current_context = ""
    for chunk in chunks:
        current_context += f"\n{chunk}"
        current_context = baseline_cpc(current_context, question, target_token)
    return current_context

def cpc_sequential_with_llmlingua(context: str, question: str, target_token: int) -> str:
    return baseline_llmlingua(baseline_cpc(context, question, target_token), question, target_token)

def cpc_with_llmlingua(context: str, question: str, target_token: int) -> str:
    return baseline_llmlingua(baseline_cpc(context, question, target_token * 2), question, target_token)

def eval_dataset(token_ranges=[1000, 2000, 3000]) -> dict:
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    dataset = dataset.filter(lambda x: x["domain"] == "Single-Document QA")
    dataset = dataset.filter(lambda x: x["length"] == "medium")

    strategies = {
        "baseline_llmlingua": baseline_llmlingua,
        "baseline_cpc": baseline_cpc,
        "cpc_sequential": cpc_sequential,
        "cpc_sequential_with_llmlingua": cpc_sequential_with_llmlingua,
        "cpc_with_llmlingua": cpc_with_llmlingua
    }

    print("Number of samples: ", len(dataset))
    results = []

    os.makedirs("results", exist_ok=True)
    results_file = "results/evaluation_results.jsonl"

    for token_count in tqdm(token_ranges, desc="Token Ranges"):
        for row in tqdm(dataset):
            question, context, answer = row["question"], row["context"], row["answer"]
            a, b, c, d = row["choice_A"], row["choice_B"], row["choice_C"], row["choice_D"]

            result = {
                "question": question,
                "context": context,
                "target": answer,
                "token_count": token_count,
                "strategies": {}
            }

            for strategy_name in strategies:
                strategy_func = strategies[strategy_name]

                start_time = time.time()
                
                compressed_version = strategy_func(context, question, token_ranges[0])
                
                compression_time = time.time()

                gpt_result = get_model_response(
                    prompt=PROMPT.format(
                        document=compressed_version,
                        question=question,
                        c_a=a,
                        c_b=b,
                        c_c=c,
                        c_d=d
                    )
                )
                model_answer = extract_response(gpt_result)

                answer_time = time.time()

                result["strategies"][strategy_name] = {
                    "compression_time": compression_time - start_time,
                    "answer_time": answer_time - compression_time,
                    "compressed_version": compressed_version,
                    "response": None,
                    "answer": None,
                    "correct": result["target"].lower() == model_answer.lower() if model_answer else None
                }
            
            results.append(result)
            with open(results_file, "a") as f:
                f.write(json.dumps(result))

if __name__ == "__main__":
    Fire(eval_dataset)