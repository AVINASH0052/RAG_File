# Document Chat RAG Application

This is a Retrieval Augmented Generation (RAG) application built with Streamlit that allows users to upload documents and chat with them using natural language.

## Features

- Upload multiple documents (PDF, DOCX, TXT)
- Process and analyze document content
- Chat interface for asking questions about the documents
- RAG-based responses using document context
- Conversation history tracking

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your HuggingFace API token:
```
HUGGINGFACE_API_KEY=your_api_key_here
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Upload your documents using the file uploader

4. Wait for the documents to be processed

5. Start asking questions about your documents in the chat interface

## Requirements

- Python 3.8 or higher
- HuggingFace API token
- Internet connection for model access

## Note

This application uses the HuggingFace Hub for embeddings and language models. Make sure you have a valid API token and sufficient credits for the models you want to use. 