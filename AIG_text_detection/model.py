import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

class TextClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)

    def train(self, train_dataset, val_dataset, epochs=1, batch_size=16, learning_rate=2e-5, save_path="best_model.pth"):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        best_accuracy = 0.0  

        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0

            for batch in tqdm(train_loader, desc="Training"):
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader):.4f}")

            accuracy = self.evaluate(val_loader)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    def evaluate(self, data_loader):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != "labels"}
                labels = batch["labels"].to(self.device)

                print(f"input_ids shape before forward: {inputs['input_ids'].shape}")
                print(f"attention_mask shape before forward: {inputs['attention_mask'].shape}")

                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy

