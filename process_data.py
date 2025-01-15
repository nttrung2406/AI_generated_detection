from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pydub import AudioSegment
import io
import torchaudio
from torchaudio.transforms import MelSpectrogram

def preprocess_audio(audio_bytes):
    try:
        audio_stream = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_stream, format="mp3")  
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        waveform, sample_rate = torchaudio.load(wav_io)

        waveform = waveform / waveform.abs().max()
        # Generate spectrogram
        spectrogram_transform = MelSpectrogram(sample_rate=sample_rate)
        spectrogram = spectrogram_transform(waveform)

        return spectrogram
    except Exception as e:
        raise RuntimeError(f"Error processing audio file: {e}")
    
def preprocess_image(image):
    preprocess = Compose([
        Resize((32, 32)),               
        ToTensor(),                     
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    return preprocess(image).unsqueeze(0)  