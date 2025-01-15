import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        print("Data loaded successfully!")
        return self.data

    def preprocess(self):
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(subset=['text'], inplace=True)
        print("Preprocessing complete!")
        return self.data

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        train, test = train_test_split(self.data, test_size=test_size, random_state=random_state, stratify=self.data['generated'])
        self.train_data, self.val_data = train_test_split(train, test_size=val_size, random_state=random_state, stratify=train['generated'])
        self.test_data = test
        print("Data split into train, validation, and test sets!")
        return self.train_data, self.val_data, test
