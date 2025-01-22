import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

# Load dataset
dataset = load_dataset("wmt17", "de-en")

# Limit the number of samples for quick training
train_samples = 1000  # Number of training samples
val_samples = 2000     # Number of validation samples
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_samples))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(val_samples))

# Load tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Preprocess the dataset
def preprocess(batch):
    inputs = [f"translate German to English: {item['de']}" for item in batch["translation"]]
    targets = [item["en"] for item in batch["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length", return_tensors="pt").input_ids
    model_inputs["labels"] = labels
    return model_inputs

processed_dataset = dataset.map(preprocess, batched=True)
processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# DataLoader
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
train_dataloader = DataLoader(processed_dataset["train"], batch_size=8, collate_fn=data_collator)
val_dataloader = DataLoader(processed_dataset["validation"], batch_size=8, collate_fn=data_collator)

# Define training function
def train_model(device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        start_time = time.time()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        end_time = time.time()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Time = {end_time - start_time:.2f}s")

# Train on CPU
print("Training on CPU")
train_model(torch.device("cpu"))

# Train on GPU if available
if torch.cuda.is_available():
    print("Training on GPU")
    train_model(torch.device("cuda"))
else:
    print("GPU not available.")
