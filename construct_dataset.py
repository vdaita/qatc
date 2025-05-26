from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import os
from fire import Fire
import json
from tqdm import tqdm

load_dotenv(".env")

def construct_dataset(
    output_dir: str = "data/qa_dataset",
    model_name: str = "google/...",
    max_questions_per_transcript: int = 5,
    total_pairs: int = 2000
):
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    dataset = load_dataset("huuuyeah/meetingbank", split="train")
    pairs = []

    for idx, item in tqdm(enumerate(dataset)):
        transcript = item["transcript"]
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate {max_questions_per_transcript} questions from the following transcript: {transcript}. Output the questions in a JSON format: {"questions": [...]}"
                }
            ],
            temperature=0.7,
            max_tokens=200
        )
        questions = response.choices[0].message.content
        questions = json.loads(questions)
        for question in questions["questions"]:
            pairs.append({
                "question": question,
                "answer": transcript
            })
        if len(pairs) >= total_pairs:
            break
    
    with open(os.path.join(output_dir, "qa_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=4)
    print(f"Dataset constructed with {len(pairs)} pairs and saved to {output_dir}/qa_pairs.json")

if __name__ == "__main__":
    Fire(construct_dataset)