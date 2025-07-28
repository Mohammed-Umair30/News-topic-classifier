from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

# Load processed test dataset
print("Loading processed test dataset...")
test_dataset = torch.load("./ag_news_processed/test_dataset.pt", weights_only=False)
print("Test dataset loaded.")

# Reduce dataset size for faster evaluation
test_dataset = test_dataset.select(range(100))   # Use first 100 samples for evaluation
print(f"Reduced test dataset size to {len(test_dataset)} samples.")

# Load fine-tuned model and tokenizer
print("Loading fine-tuned model and tokenizer...")
model_dir = "./fine_tuned_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
print("Fine-tuned model and tokenizer loaded.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Create DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=8)

# Evaluate the model
print("Starting model evaluation...")
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["labels"].cpu().numpy())

# Calculate accuracy and F1-score
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')

print("Model evaluation complete.")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score (weighted): {f1:.4f}")

# Save evaluation results
results = {
    "accuracy": accuracy,
    "f1_score": f1,
    "num_samples": len(test_dataset)
}

print(f"Evaluation Results:")
print(f"- Accuracy: {accuracy:.4f}")
print(f"- F1-score (weighted): {f1:.4f}")
print(f"- Number of test samples: {len(test_dataset)}")

# AG News class labels
class_labels = ["World", "Sports", "Business", "Sci/Tech"]
print(f"Class labels: {class_labels}")

