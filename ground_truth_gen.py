from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import os
from fire import Fire
import json
from tqdm import tqdm
from label_tokens import label_chunk

SYSTEM_PROMPT = """
Compress the given text to short expressions, and such that you (an expert LLM) understand the relevant context within the text itself and answer the question. Unlike the usual text compression, I need you to comply with the 5 conditions below:
1. You can ONLY remove unimportant words. 2. Do not reorder the original words.
3. Do not change the original words.
4. Do not use abbreviations or emojis.
5. Do not add new words or symbols.
Compress the origin aggressively by removing words only. Compress the origin as short as you can, while retain- ing as much information as possible.
If you can, please compress the following text based on the provided question: 
"""

FORMATTED_PROMPT = """
<context>
{context}
</context>
<question>
{question}
</question>
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

    token_data = {
        "labels": [],
        "origin": [],
        "comp": [],
        "retrieval": [],
        "origin_tokens": [],
        "comp_rate": [],
        "variation_rate": [],
        "hitting_rate": [],
        "matching_rate": [],
        "alignment_gap": []
    }

    for qa in tqdm(input_qa):
        question, answer = qa["question"], qa["answer"]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": FORMATTED_PROMPT.format(question=question, answer=answer) + "\n"
                }
            ],
            temperature=0.7,
            max_tokens=max_chunk_tokens
        )
        response = response.choices[0].message.content
        qa["ground_truth"] = response

        qa_labeled = label_chunk(
            origin=question,
            comp=response,
            window_size=max_chunk_tokens
        )
        for key in qa_labeled:
            token_data[key].append(qa_labeled[key])
    
    output_data = {
        "token_data": token_data,
        "raw_text": input_qa 
    }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Ground truth dataset constructed with {len(input_qa)} pairs and saved to {output_json}")

if __name__ == "__main__":
    load_dotenv(".env")
    Fire(construct_ground_truth_dataset)