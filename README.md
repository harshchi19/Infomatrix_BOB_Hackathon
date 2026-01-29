# Azure AI Services Demo - Bank of Baroda Hackathon

A comprehensive Streamlit-based web application that demonstrates the power of Azure AI services including Language Processing, Computer Vision, Speech Services, and OpenAI integration. Built for the Infomatrix Bank of Baroda Hackathon.

## 🚀 Features

This application provides a unified interface to interact with multiple Azure AI services:

### 1. **Language Services**
- **Sentiment Analysis**: Analyze text sentiment with confidence scores (positive, neutral, negative)
- Powered by Azure Text Analytics

### 2. **Computer Vision**
- **Image Analysis**: Analyze images using direct URLs
- **Description Generation**: Automatic image description
- **Tag Extraction**: Identify and extract relevant tags from images
- Supports JPG, JPEG, and PNG formats

### 3. **Speech Services**
- **Text-to-Speech (TTS)**: Convert text to natural-sounding speech using Azure Neural voices
- **Speech-to-Text (STT)**: Real-time speech recognition and transcription
- Uses Azure Speech SDK with Jenny Neural voice (en-US)

### 4. **OpenAI Services**
- **Text Generation**: Generate text using Azure OpenAI models
- **PDF Processing**: Upload and analyze PDF documents with Q&A capabilities
- **Speech Integration**: Convert inputs and outputs to/from speech
- Customizable parameters: max tokens, temperature, and top_p
- Support for multiple input types: text, PDF, and speech

## 📁 Project Structure

```
Infomatrix_BOB_Hackathon/
├── app.py                      # Main application with service selector
├── app1.py                     # Simplified version with image upload
├── updated_app                 # Enhanced version with PDF & speech integration
├── Azure Doc Intelligence/     # Document analysis workflow using Prompt Flow
│   ├── flow.dag.yaml          # Workflow definition
│   ├── requirements.txt       # Dependencies for document intelligence
│   ├── create_document.py     # Document creation utilities
│   ├── parse_skill_to_text.py # Text parsing functions
│   └── read_file.py           # File reading utilities
├── Supporting Diagrams/       # Supporting documentation and diagrams
└── README.md                  # This file
```

## 🛠️ Prerequisites

- Python 3.8 or higher
- Azure subscription with the following services:
  - Azure Cognitive Services (Language & Vision)
  - Azure Speech Services
  - Azure OpenAI Service
  - Azure AI Translator (optional, for document intelligence flow)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/harshchi19/Infomatrix_BOB_Hackathon.git
   cd Infomatrix_BOB_Hackathon
   ```

2. **Install required dependencies**:
   ```bash
   pip install streamlit requests azure-cognitiveservices-vision-computervision msrest azure-ai-textanalytics azure-cognitiveservices-speech PyPDF2 pydub pyperclip
   ```

3. **For Azure Document Intelligence workflow** (optional):
   ```bash
   cd "Azure Doc Intelligence"
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

Before running the application, you need to configure your Azure service credentials:

### Option 1: Update credentials in the code (for testing)
Edit the application file (`app.py`, `app1.py`, or `updated_app`) and update the following variables:
```python
KEY = "your-azure-cognitive-services-key"
OPENAI_ENDPOINT = "your-azure-openai-endpoint"
COGNITIVE_ENDPOINT = "your-cognitive-services-endpoint"
SPEECH_KEY = "your-speech-services-key"
SPEECH_REGION = "your-region"  # e.g., "eastus"
```

### Option 2: Use environment variables (recommended for production)
```bash
export AZURE_COGNITIVE_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_COGNITIVE_ENDPOINT="your-endpoint"
export AZURE_SPEECH_KEY="your-key"
export AZURE_SPEECH_REGION="your-region"
```

**⚠️ Security Note**: Never commit API keys to version control. Use environment variables or Azure Key Vault for production deployments.

## 🎯 Usage

### Running the Main Application

Choose one of the following versions based on your needs:

1. **Full-featured version with service selector** (`app.py`):
   ```bash
   streamlit run app.py
   ```

2. **Simplified version with image upload** (`app1.py`):
   ```bash
   streamlit run app1.py
   ```

3. **Enhanced version with PDF and speech integration** (`updated_app`):
   ```bash
   streamlit run updated_app
   ```

The application will open in your default web browser at `http://localhost:8501`.

### Using Different Services

#### Language Services
1. Select "Language" from the sidebar
2. Enter text for sentiment analysis
3. Click "Analyze" to see sentiment results with confidence scores

#### Computer Vision
1. Select "Vision" from the sidebar
2. Enter a direct image URL (must end with .jpg, .png, or .jpeg)
3. Click "Analyze Image" to see descriptions and tags

#### Speech Services
1. Select "Speech" from the sidebar
2. Enter text to convert to speech
3. Click "Convert to Speech" to hear the audio output

#### OpenAI Services
1. Select "OpenAI" from the sidebar
2. Enter your deployment name
3. Choose input type:
   - **Text**: Enter a prompt directly
   - **PDF**: Upload a PDF and ask questions about it
   - **Speech to Text**: Use microphone for voice input
4. Adjust generation parameters (max tokens, temperature, top_p)
5. Click "Generate" to see results
6. Optionally convert output to speech

### Azure Document Intelligence Workflow

The `Azure Doc Intelligence` folder contains a Prompt Flow for advanced document analysis:

```bash
cd "Azure Doc Intelligence"
pf flow test --flow .
```

This workflow provides:
- Document translation
- PII detection and redaction
- Named Entity Recognition (NER)
- Document summarization (extractive & abstractive)
- Sentiment analysis and opinion mining

## 🔧 Technologies Used

- **Streamlit**: Web application framework
- **Azure Cognitive Services**: Language and vision AI capabilities
- **Azure Speech SDK**: Speech synthesis and recognition
- **Azure OpenAI**: Advanced language models
- **Azure AI Language**: Document analysis and NLP
- **Prompt Flow**: Document processing workflows
- **PyPDF2**: PDF processing
- **Python 3.8+**: Core programming language

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project was created for the Infomatrix Bank of Baroda Hackathon.

## 👥 Authors

- Harsh Chitraksh ([@harshchi19](https://github.com/harshchi19))

## 🙏 Acknowledgments

- Microsoft Azure for providing comprehensive AI services
- Bank of Baroda for hosting the Infomatrix Hackathon
- Streamlit for the intuitive web framework

## 📞 Support

For questions or issues, please open an issue in the GitHub repository or contact the maintainers.

---

**Note**: This project contains API keys for demonstration purposes. Please replace them with your own keys before deploying to production. Never commit sensitive credentials to version control.