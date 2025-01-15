import streamlit as st
import torch
import os
from model_loader import load_text_model, load_voice_model, load_image_model
from detector import detect_aig_text, detect_aig_image, detect_aig_voice

# Functions for detection
st.sidebar.title("AI Detection")
options = ["Text", "Image", "Voice"]
choice = st.sidebar.radio("Select a function:", options)

# Text Detection
tokenizer, text_model = load_text_model()
if choice == "Text":
    st.title("AI-Generated Text Detection")
    text_input = st.text_area("Enter text to detect:")
    if st.button("Detect"):
        if text_input:
            probability = detect_aig_text(text_input, tokenizer, text_model)
            if probability >= 0.8:
                st.write("Your text is probably generated with AI")
            else:
                st.write("Your text is probably written by human")
            st.write(f"Probability of being AI-generated: {probability:.2%}")
        else:
            st.warning("Please enter some text.")

# Image Detection
elif choice == "Image":
    st.title("AI-Generated Image Detection")
    image_model = load_image_model()
    uploaded_image = st.file_uploader("Upload an image for detection:", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        if st.button("Detect"):
            probability = detect_aig_image(image_bytes, image_model)
            if probability >= 0.8:
                st.write("Your image is probably generated with AI")
            else:
                st.write("Your image is probably real")
            st.write(f"Probability of being AI-generated: {probability:.2%}")

# Voice Detection
elif choice == "Voice":
    st.title("AI-Generated Voice Detection")
    voice_model = load_voice_model()
    uploaded_audio = st.file_uploader("Upload an audio file for detection:", type=["wav", "mp3"])
    print(f"Uploaded file type: {uploaded_audio.type}") 
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        if st.button("Detect"):
            probability = detect_aig_voice(audio_bytes, voice_model)
            if probability > 0.5:
                st.write("AI-generated voice detected with probability:", round(probability, 2))
            else:
                st.write("Human voice detected with probability:", round(1 - probability, 2))
                         
    elif uploaded_audio is None:
        st.warning("Please upload an audio file.")
