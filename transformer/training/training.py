import torch
import torch.optim as optim
from modelling.transformer import TransformerModel
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset


class LearningRateScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        """
        Initialisiert den Scheduler.
        :param optimizer: Der Optimierer, dessen Lernrate angepasst wird.
        :param d_model: Die Modellgröße (dimensionsabhängige Skalierung).
        :param warmup_steps: Die Anzahl der Warmup-Schritte.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0  # Aktueller Schritt

    def step(self):
        """
        Aktualisiert die Lernrate basierend auf der aktuellen Schrittzahl.
        """
        self.step_num += 1
        lr = self.calculate_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_lr(self):
        """
        Berechnet die Lernrate basierend auf der Formel.
        :return: Die berechnete Lernrate.
        """
        scale = self.d_model ** -0.5
        lr = scale * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        return lr

    def get_lr(self):
        """
        Gibt die aktuelle Lernrate zurück.
        :return: Die aktuelle Lernrate.
        """
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def test_lrs():
    # Beispiel-Parameter
    d_model = 512  # Modellgröße
    warmup_steps = 4000  # Warmup-Schritte
    model = torch.nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.0)

    # Scheduler initialisieren
    scheduler = LearningRateScheduler(optimizer, d_model, warmup_steps)

    # Trainingsloop
    for epoch in range(10):  # Beispielhafte 10 Epochen
        for batch in range(100):  # Beispielhafte 100 Batches pro Epoche
            # (Hier würde das Modell trainiert werden)
            scheduler.step()  # Lernrate aktualisieren
            print(f"Step {scheduler.step_num}, Learning Rate: {scheduler.get_lr()[0]:.6f}")


# Define parameter groups for the optimizer
def get_optimizer(model, learning_rate, weight_decay):
    """
    Returns an AdamW optimizer with parameter groups ensuring no weight decay
    is applied to bias and LayerNorm parameters.

    Args:
    - model (nn.Module): The Transformer model.
    - learning_rate (float): Learning rate for the optimizer.
    - weight_decay (float): Weight decay coefficient.

    Returns:
    - torch.optim.AdamW: Configured AdamW optimizer.
    """
    # Separate parameters into those with and without weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:  # Only include trainable parameters
            if "bias" in name or "LayerNorm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # Create parameter groups
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Initialize AdamW optimizer
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer


# Example usage
def test_AdamW():
    model = TransformerModel(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=100,
    )

    # Define hyperparameters
    learning_rate = 5e-4
    weight_decay = 0.01

    # Initialize the optimizer
    optimizer = get_optimizer(model, learning_rate, weight_decay)

    # Print optimizer parameter groups
    print(optimizer)


def preprocess_with_special_tokens(dataset, tokenizer, max_length=128):
    """
    Preprocess the dataset by tokenizing and baking in special tokens.

    Args:
    - dataset: The dataset to preprocess.
    - tokenizer: The tokenizer to use.
    - max_length (int): Maximum sequence length for padding and truncation.

    Returns:
    - Preprocessed dataset with baked-in special tokens.
    """
    def process_example(example):
        # Add special tokens to the target translations
        example["translation"]["en"] = f"{tokenizer.bos_token} {example['translation']['en']} {tokenizer.eos_token}"
        return example

    # Map the process_example function over the dataset
    dataset = dataset.map(process_example)

    # Tokenize inputs and targets
    def tokenize_batch(batch):
        inputs = [ex["de"] for ex in batch["translation"]]
        targets = [ex["en"] for ex in batch["translation"]]

        # Tokenize the inputs and targets
        tokenized_inputs = tokenizer(
            inputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        tokenized_targets = tokenizer(
            targets, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_targets["input_ids"],
        }

    # Apply the tokenization
    dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["translation"])

    return dataset



def train():
    # Load dataset and tokenizer
    dataset = load_dataset("wmt/wmt17", "de-en")

    # Reduce size for testing
    train_dataset = dataset["train"].shuffle(seed=32).select(range(100000))  # 3,000 samples
    val_dataset = dataset["validation"]

    # Load and update tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    tokenizer.add_special_tokens({"bos_token": "<start>", "eos_token": "<end>"})

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Special tokens added: BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}")

    # Initialize model and resize embeddings
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=128,
        dropout=0.1,
        max_len=128,
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model embedding size after resizing: {model.embedding.weight.size()}")

    # Preprocess datasets with resized tokenizer
    train_dataset = preprocess_with_special_tokens(train_dataset, tokenizer)
    val_dataset = preprocess_with_special_tokens(val_dataset, tokenizer)

    # Debug: Print a sample preprocessed data entry
    print(f"Sample preprocessed train data: {train_dataset[0]}")

    # Data collator and DataLoaders
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)

    # Optimizer, scheduler, and criterion
    optimizer = get_optimizer(model, learning_rate=1e-4, weight_decay=0.1)
    scheduler = LearningRateScheduler(optimizer, d_model=64, warmup_steps=400)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float('inf')
    patience = 5
    no_improve_epochs = 0
    min_delta = 0.001  # Minimum change in validation loss

    for epoch in range(5):
        print(f"Starting epoch {epoch + 1}...")

        # Training loop
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Model forward pass
            outputs = model(src=input_ids, tgt=labels, src_mask=attention_mask, tgt_mask=None)
            logits = outputs

            # Compute loss and backpropagate
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(src=input_ids, tgt=labels, src_mask=attention_mask, tgt_mask=None)
                logits = outputs
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}")

        # Early Stopping Check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            # Save the model
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print("Model saved!")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return model



if __name__ == "__main__":
    trained_model = train()

    # Save the trained model weights outside the training function
    torch.save(trained_model.state_dict(), "trained_transformer_model.pth")
    print("Model weights saved to 'trained_transformer_model.pth'")
