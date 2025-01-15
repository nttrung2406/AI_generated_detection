from data_processing import DatasetHandler
from feature_extractor import FeatureExtractor
from model import TextClassifier
from data_reader import TextDataset
from torch.utils.data import DataLoader
import os

class Pipeline:
    def __init__(self, filepath, model_name="distilbert-base-uncased"):
        self.dataset_handler = DatasetHandler(filepath)
        self.feature_extractor = FeatureExtractor(model_name)
        self.text_classifier = TextClassifier(model_name)

    def run(self):
        # Step 1: Load and preprocess data
        print("Loading and preprocessing data...")
        data = self.dataset_handler.load_data()
        data = self.dataset_handler.preprocess()
        
        # Step 2: Split data
        print("Splitting data into train, validation, and test sets...")
        train_data, val_data, test_data = self.dataset_handler.split_data()

        # Step 3: Tokenize data
        print("Tokenizing data...")
        tokenizer = self.feature_extractor.tokenizer  # Extract tokenizer

        train_dataset = TextDataset(
            texts=train_data['text'].tolist(),
            labels=train_data['generated'].tolist(),
            tokenizer=tokenizer
        )
        val_dataset = TextDataset(
            texts=val_data['text'].tolist(),
            labels=val_data['generated'].tolist(),
            tokenizer=tokenizer
        )
        test_dataset = TextDataset(
            texts=test_data['text'].tolist(),
            labels=test_data['generated'].tolist(),
            tokenizer=tokenizer
        )

        # Debug shapes of the test dataset
        print("Debugging DataLoader shapes for test dataset...")
        test_loader = DataLoader(test_dataset, batch_size=16)
        for batch in test_loader:
            print(f"input_ids shape: {batch['input_ids'].shape}")
            print(f"attention_mask shape: {batch['attention_mask'].shape}")
            print(f"labels shape: {batch['labels'].shape}")
            break 

        # Step 4: Train the model
        print("Training the model...")
        self.text_classifier.train(train_dataset, val_dataset, epochs=1, batch_size=16, save_path="best_model.pth")

        # Step 5: Evaluate the model
        print("Evaluating the model...")
        test_loader = DataLoader(test_dataset, batch_size=16)
        self.text_classifier.evaluate(test_loader)

if __name__ == "__main__":
    root = os.getcwd()
    train_path = os.path.join(root, 'train.csv')
    
    pipeline = Pipeline(filepath=train_path)
    pipeline.run()
