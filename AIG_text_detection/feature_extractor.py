from transformers import AutoTokenizer

class FeatureExtractor:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, texts, max_length=512):
        return self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
