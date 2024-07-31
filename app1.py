import streamlit as st
import requests
import json
import base64
import os
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk
import io
import tempfile
import time

KEY = "8019113bcf564d2c8236f3a8ef8865f0"
OPENAI_ENDPOINT = "https://prnvpwr2612bobtrial2502024.openai.azure.com/"
COGNITIVE_ENDPOINT = "https://prnvpwr2612bobtrial2502024.cognitiveservices.azure.com/"
TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"
STT_ENDPOINT = "https://eastus.stt.speech.microsoft.com"
TTS_ENDPOINT = "https://eastus.tts.speech.microsoft.com"

st.set_page_config(page_title="Azure AI Demo", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #4e54c8, #8f94fb);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Azure AI Services Demo")

# Language Services
st.header("Language Services")
text_input = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    try:
        text_analytics_client = TextAnalyticsClient(endpoint=COGNITIVE_ENDPOINT, credential=AzureKeyCredential(KEY))
        result = text_analytics_client.analyze_sentiment(documents=[text_input])[0]
        st.write(f"Sentiment: {result.sentiment}")
        st.write(f"Confidence scores: Positive={result.confidence_scores.positive:.2f}, Neutral={result.confidence_scores.neutral:.2f}, Negative={result.confidence_scores.negative:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Vision Services
st.header("Computer Vision")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        cv_client = ComputerVisionClient(COGNITIVE_ENDPOINT, CognitiveServicesCredentials(KEY))
        features = [VisualFeatureTypes.description, VisualFeatureTypes.tags]
        results = cv_client.analyze_image_in_stream(io.BytesIO(image_bytes), visual_features=features)
        st.image(image_bytes, caption="Analyzed Image", use_column_width=True)
        st.write("Description:", results.description.captions[0].text)
        st.write("Tags:", ", ".join([tag.name for tag in results.tags]))
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Speech Services
st.header("Speech Services")
text_to_speak = st.text_input("Enter text to convert to speech:")
if st.button("Convert to Speech"):
    temp_filename = None
    try:
        speech_config = speechsdk.SpeechConfig(subscription=KEY, region="eastus")
        speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            temp_filename = tmp_file.name

        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_filename)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = synthesizer.speak_text_async(text_to_speak).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            with open(temp_filename, "rb") as audio_file:
                audio_data = audio_file.read()
            st.audio(audio_data, format="audio/wav")
            st.success("Speech synthesized successfully!")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            st.error(f"Speech synthesis canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                st.error(f"Error details: {cancellation_details.error_details}")
        else:
            st.error(f"Speech synthesis failed: {result.reason}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if temp_filename and os.path.exists(temp_filename):
            for _ in range(5):  
                try:
                    os.remove(temp_filename)
                    break
                except PermissionError:
                    time.sleep(1)  
            else:
                st.warning("Could not delete temporary file. It will be removed automatically later.")

# OpenAI Services
st.header("OpenAI Services")
OPENAI_ENDPOINT = "https://bobtrialinfomatrix.openai.azure.com/"
KEY = "37db37e7aff6444fa6f59fcb1fc48028"

st.info("""
Please enter your Azure OpenAI deployment name. This is not your resource name, but the name you gave to your specific model deployment.
You can find this in the Azure portal under your OpenAI resource > Deployments.
""")

DEPLOYMENT_NAME = st.text_input("Enter your Azure OpenAI deployment name:")
prompt = st.text_area("Enter a prompt for text generation:", height=150)

col1, col2, col3 = st.columns(3)
with col1:
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=2000, value=500, step=50)
with col2:
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
with col3:
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

if st.button("Generate") and DEPLOYMENT_NAME:
    headers = {
        "Content-Type": "application/json",
        "api-key": KEY
    }
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "best_of": 1,
        "stop": None
    }
    try:
        with st.spinner("Generating text..."):
            response = requests.post(f"{OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/completions?api-version=2023-05-15", headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            generated_text = result['choices'][0]['text']
            st.subheader("Generated Text:")
            st.write(generated_text)
            
            st.text_area("Copy generated text:", value=generated_text, height=300)
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        if response.status_code == 404:
            st.error("Deployment not found. Please check your deployment name and try again.")
        else:
            st.error(f"Detailed error: {response.text}")

st.sidebar.info("This is a demo of various Azure AI services.")
