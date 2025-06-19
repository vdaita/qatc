from openai import OpenAI
from transformers import AutoModelForTokenClassification, pipeline
from tokenizers import AutoTokenizer
from fire import Fire
from datasets import Dataset, load_dataset
import json
from tqdm import tqdm
from label_tokens import split_string
from openai import OpenAI
import os
from ground_truth_gen import FORMATTED_PROMPT
import torch

MAX_STRING_LENGTH = 200

def main(model_dir: str, output_file: str = "results/generated_responses.json"):
    longbench_dataset = load_dataset("THUDM/LongBench-v2")
    longbench_dataset = longbench_dataset.filter(lambda x: x["domain"] == "Multi-Document QA")

    client = OpenAI(
        model="gpt-3.5-turbo-0125",
        api_key=os.getenv("OPENAI_APIP_KEY")
    )
    classifier = pipeline("ner", model=model_dir)

    for row in longbench_dataset:
        question, context, answer = row["question"], row["context"], row["answer"]
        a, b, c, d = row["choice_A"], row["choice_B"], row["choice_C"], row["choice_D"]

        split_context = split_string(context)
        context_chunks = [split_context[i : max(i + MAX_STRING_LENGTH, len(split_context))] for i in range(0, len(split_context), MAX_STRING_LENGTH)]
        context_chunks = [" ".join(chunk) for chunk in context_chunks]

        context_chunks = [FORMATTED_PROMPT.format(context=chunk, question=question) for chunk in context_chunks]
        classified_chunks = classifier(context_chunks)
        texts = []
        
        for text in classified_chunks:
            truncated_text = ""
            for entity in text:
                if entity["entity_group"] == "INCL":
                    truncated_text += entity["word"] + " "
            texts.append(truncated_text.strip())
        