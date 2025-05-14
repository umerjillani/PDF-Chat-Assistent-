# PDF RAG System with Google Drive Integration

A powerful Retrieval-Augmented Generation (RAG) system that enables natural language querying of PDF documents stored in Google Drive. Built with LangChain, OpenAI, FAISS, and Flask, this system provides an intelligent document search and Q&A interface.

## Features

- 🔄 Automatic Google Drive document synchronization
- 📄 PDF text extraction with PyMuPDF
- 🔍 OCR support for image-based PDFs (Tesseract/EasyOCR)
- 🧠 Conversational memory for context-aware responses
- 🔢 FAISS vector similarity search
- 🌐 REST API for Custom GPT integration
- 📊 Document processing status mo

## Prerequisites

- Python 3.8+
- OpenAI API key
- Google Cloud project with Drive API enabled
- Service account credentials for Google Drive
- Tesseract OCR (optional, for OCR support)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following configuration:
```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-3.5-turbo
TEMPERATURE=0.3
CHUNK_SIZE=1024
CHUNK_OVERLAP=100
OCR_ENABLED=True
TESSERACT_LANG=eng
MAX_CONCURRENT_OCR=4
GOOGLE_SERVICE_ACCOUNT_FILE=service-account.json
GOOGLE_DRIVE_FOLDER_ID=your_google_drive_folder_id
```

## Google Drive Setup

1. **Enable Google Drive API**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing one
   - Navigate to APIs & Services > Library
   - Search for "Google Drive API" and enable it

2. **Create Service Account**
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "Service Account"
   - Fill in service account details
   - Assign the Editor role
   - Create and download JSON key file
   - Rename to `service-account.json` and place in project root

3. **Configure Google Drive**
   - Share your target Google Drive folder with the service account email
   - Get the folder ID from the URL: `https://drive.google.com/drive/folders/[FOLDER_ID]`
   - Add the folder ID to your `.env` file

## Running the System

1. Start the Flask server:
```bash
python server.py
```

The server will run on `http://localhost:8000` by default.

## API Endpoints

- `POST /query` - Submit a question about your documents
- `GET /status` - Check system status and sync state
- `POST /sync` - Trigger manual document synchronization
- `POST /reset` - Reset conversation memory

## Custom GPT Integration

1. **Expose Local Server**
   - Install ngrok: `npm install -g ngrok`
   - Run: `ngrok http 8000`
   - Copy the HTTPS URL provided by ngrok

2. **Create Custom GPT**
   - Visit [ChatGPT GPT Creation](https://chat.openai.com/gpts)
   - Click "Create a GPT"
   - Configure with the following settings:

   **Name**: PDF Chat Assistant
   
   **Description**: A document assistant powered by RAG technology that answers questions based on your synchronized documents.

   **Instructions**:
   ```
   You are a document assistant powered by a Retrieval-Augmented Generation (RAG) backend. Your job is to collect a user's query and send it to the RAG API endpoint /query to get an accurate answer based on synchronized documents.

   When the user asks a question related to any document, legal case, policy, or reference material, call the askQuestion action with the user's exact query as the question parameter.

   Do not try to answer on your own. Only respond with the answer field returned by the API.

   If the user says something casual (like "hi", "hello", "how are you", "tell me a joke", etc.), politely respond with:
   "Please ask a question related to your documents so I can help you accurately."

   If the system is syncing, inform the user that it's currently syncing and to try again in a few minutes.

   If the user says something like "upload new documents", "resync files", "process new docs", or anything similar, call the syncDocuments operation to sync and ingest the latest documents from Google Drive. Notify the user once syncing is complete or if an error occurs.

   If the user says things like "check system status" or "is it ready?", or anything similar call getStatus.

   If the user says "clear chat", "reset memory", or similar, call resetMemory.
   ```

3. **Configure Actions**
   - Set Auth Type to: None
   - Update the OpenAPI schema with your ngrok URL
   - Add the following actions:
     - askQuestion
     - syncDocuments
     - getStatus
     - resetMemory

## System Architecture

The system consists of two main components:

1. **Rag_system.py**
   - Document processing pipeline
   - OCR handling
   - Vector store management
   - Query processing

2. **server.py**
   - Flask API endpoints
   - Google Drive synchronization
   - Status monitoring
   - Memory management

## Directory Structure

```
.
├── Rag_system.py          # Main RAG processing logic
├── server.py             # Flask API server
├── service-account.json  # Google Drive credentials
├── .env                  # Environment configuration
├── data/                 # Downloaded PDF storage
├── faiss_index/         # Vector store directory
└── requirements.txt      # Python dependencies
```

## Error Handling

The system includes comprehensive error handling for:
- API key validation
- Google Drive authentication
- Document processing
- OCR operations
- Vector store management
- API request validation

