from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
import os
from fire import Fire
import json
from tqdm import tqdm
from transformers import AutoTokenizer

STRING_FORMAT = """<content>
{content}
</content>
<question>
{question}
</question>
"""

def split_into_chunks(text: str, tokenizer: AutoTokenizer, chunk_size: int):
    tokens = tokenizer.encode(text)    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def construct_dataset(
    model_name: str = "openai/gpt-4.1-mini",
    tokenizer_name: str = "answerdotai/ModernBERT-base",
    max_questions_per_transcript: int = 5,
    max_examples: int = 30000
):
    
    print("OpenRouter API Key:", os.environ.get("OPENROUTER_API_KEY"))
        
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    dataset = load_dataset("huuuyeah/meetingbank", split="train")
    examples = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name) 

    if not os.path.exists("../results/meetingbank/"):
        os.makedirs("../results/meetingbank/")

    for idx, item in tqdm(enumerate(dataset)):
        try:
            transcript = item["transcript"]
            chunked_transcript = split_into_chunks(transcript, tokenizer, 1024)

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": f"Generate {max_questions_per_transcript} questions from the following transcript: {transcript}. Output the questions in a JSON format: {{\"questions\": []}}"
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=400
            )
            questions = response.choices[0].message.content

            questions = json.loads(questions)
            for question in questions["questions"]:
                for chunk in chunked_transcript:
                    examples.append(STRING_FORMAT.format(
                        content=chunk,
                        question=question
                    ))

            if idx % 20 == 0:
                with open(os.path.join("../results/meetingbank/", "examples.json"), "w") as f:
                    json.dump(examples, f, indent=4)
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue

        if len(examples) >= max_examples:
            break

    print(f"Dataset constructed with {len(examples)} pairs and saved to ../results/meetingbank/qa_pairs.json")

if __name__ == "__main__":
    Fire(construct_dataset)