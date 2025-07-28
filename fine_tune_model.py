from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os

# Load processed datasets
print("Loading processed datasets...")
train_dataset = torch.load("./ag_news_processed/train_dataset.pt", weights_only=False)
test_dataset = torch.load("./ag_news_processed/test_dataset.pt", weights_only=False)
print("Datasets loaded.")

# Reduce dataset size for faster execution
train_dataset = train_dataset.select(range(100)) # Use first 100 samples
test_dataset = test_dataset.select(range(20))   # Use first 20 samples
print(f"Reduced train dataset size to {len(train_dataset)} samples.")
print(f"Reduced test dataset size to {len(test_dataset)} samples.")

# Load pre-trained DistilBERT model and tokenizer
print("Loading pre-trained DistilBERT model and tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
print("DistilBERT model and tokenizer loaded.")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2) # Further reduced batch size
test_dataloader = DataLoader(test_dataset, batch_size=2) # Further reduced batch size

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) # Smaller learning rate
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the model (manual training loop)
print("Starting model fine-tuning...")
num_epochs = 1 # One epoch for faster execution

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            print(f"RuntimeError during training: {e}")
            print("Skipping this batch.")
            optimizer.zero_grad() # Clear gradients to prevent accumulation
            continue

print("Model fine-tuning complete.")

# Save the fine-tuned model
output_model_dir = "./fine_tuned_bert_model"
os.makedirs(output_model_dir, exist_ok=True)
model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Fine-tuned model and tokenizer saved to {output_model_dir}")


