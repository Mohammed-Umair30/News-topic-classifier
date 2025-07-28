from datasets import load_dataset
from transformers import AutoTokenizer
import os
import torch

# Load the AG News dataset
print("Loading AG News dataset...")
dataset = load_dataset("ag_news")
print("Dataset loaded.")

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # Use DistilBERT tokenizer
print("Tokenizer loaded.")

def tokenize_function(examples):
    # DistilBERT does not use token_type_ids, so we don\"t include it
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=32) # Further reduced max_length

# Apply tokenization to the dataset
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
print("Dataset tokenized.")

# Rename columns and set format for PyTorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Create a directory to save the processed data
output_dir = "./ag_news_processed"
os.makedirs(output_dir, exist_ok=True)

# Save the processed datasets
print("Saving processed datasets...")
torch.save(tokenized_datasets["train"], os.path.join(output_dir, "train_dataset.pt"))
torch.save(tokenized_datasets["test"], os.path.join(output_dir, "test_dataset.pt"))
print(f"Processed datasets saved to {output_dir}")


