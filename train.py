import json
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

def tokenize_and_align_labels(examples):
    ...

def train_model(
    model_name: str = "answerdotai/ModernBERT-base-uncased",
    train_file: str = "data/qa_dataset/ground_truth.json"
):
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    training_args = TrainingArguments(

    )