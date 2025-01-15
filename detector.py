from process_data import preprocess_audio, preprocess_image
import torch
from PIL import Image
import io
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def detect_aig_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  

def detect_aig_voice(audio_data, model):
    spectrogram = preprocess_audio(audio_data).to(device)
    print("spectrogram: ", spectrogram.shape)
    spectrogram = spectrogram.unsqueeze(0)
    print("Spectrogram shape after adding batch dimension: ", spectrogram.shape)
    model.to(device)
    outputs = model(spectrogram)
    probs = outputs.item()
    return probs


def detect_aig_image(image_data, model):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")  
        input_tensor = preprocess_image(image).to(device)
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=-1) 
        return probs[0][0].item()  
    except Exception as e:
        raise RuntimeError(f"Error processing image file: {e}")