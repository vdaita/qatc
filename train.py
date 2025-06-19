import json
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from tokenizers import AutoTokenizer
from train_utils import TokenClfDataset
from fire import Fire
from datasets import Dataset

def train_model(
    model_name: str = "answerdotai/ModernBERT-base-uncased",
    data_file: str = "data/qa_dataset/ground_truth.json"
):
    id2label = {0: "EXCL", 1: "INCL"}
    label2id = {"EXCL": 0, "INCL": 1}

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = json.load(open(data_file, "r"))

    texts = data["token_data"]["origin"]
    labels = data["token_data"]["labels"]
    dataset = TokenClfDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        model_name=model_name
    )

    num_train = (int) (len(texts) * 0.8)

    hf_dataset = {
        "tokens": [],
        "labels": []
    }
    for i in range(len(texts)):
        tokenized_text_i, labels_i = dataset[i]
        hf_dataset["tokens"].append(tokenized_text_i)
        hf_dataset["labels"].append(labels_i)

    train_dataset = {"tokens": hf_dataset["tokens"][:num_train], "labels": hf_dataset["labels"][:num_train]}
    test_dataset = {"tokens": hf_dataset["tokens"][num_train:], "labels": hf_dataset["labels"][num_train:]}
    train_dataset, test_dataset = Dataset.from_dict(train_dataset), Dataset.from_dict(test_dataset)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="output",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        ),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        preprocessing_class=tokenizer,
    )

    trainer.train()
    
if __name__ == "__main__":
    Fire(train_model)