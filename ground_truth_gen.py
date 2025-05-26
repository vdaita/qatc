from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import os
from fire import Fire
import json
from tqdm import tqdm
from label_tokens import label_chunk

PROMPT = """
"""

def construct_ground_truth_dataset(
    input_qa_json: str = "data/qa_dataset/qa_pairs.json",
    output_json: str = "data/qa_dataset/ground_truth.json",
    model_name: str = "google/...",    
    max_chunk_tokens: int = 512
):
    input_qa = json.load(open(input_qa_json, "r"))
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    for qa in tqdm(input_qa):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(question=qa["question"], answer=qa["answer"])
                }
            ],
            temperature=0.7,
            max_tokens=max_chunk_tokens
        )
        response = response.choices[0].message.content
        qa["ground_truth"] = response
    
    with open(output_json, "w") as f:
        json.dump(input_qa, f, indent=4)
    print(f"Ground truth dataset constructed with {len(input_qa)} pairs and saved to {output_json}")

if __name__ == "__main__":
    load_dotenv(".env")
    Fire(construct_ground_truth_dataset)