import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
from AIG_voice_detection.model import CNNNetwork
from AIG_image_detection.model_utils import ResNetClassifier

root = os.getcwd()
print("Root: ", root)
AIG_text_path = os.path.join(root, 'AIG_text_detection', 'best_model.pth')
AIG_voice_path = os.path.join(root, 'AIG_voice_detection', 'best_model.pth')
AIG_image_path = os.path.join(root, 'AIG_image_detection', 'best_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_text_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    print("Call model successfully")
    model.load_state_dict(torch.load(AIG_text_path, map_location=device))
    print("Model successfully loaded")
    model.to(device)
    return tokenizer, model

@st.cache_resource
def load_voice_model():
    model = CNNNetwork()
    state_dict = torch.load(AIG_voice_path, map_location=device, weights_only=True)
    # Remove unexpected keys
    state_dict.pop("linear.weight", None)
    state_dict.pop("linear.bias", None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_image_model():
    num_classes = 2  
    model = ResNetClassifier(num_classes=num_classes) 
    model.load_state_dict(torch.load(AIG_image_path, map_location=device))
    model.to(device)
    model.eval()
    return model