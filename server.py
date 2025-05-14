from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import logging
from dotenv import load_dotenv
import os
import sys
from langchain.memory import ConversationBufferMemory
from Rag_system import (
    check_openai_api_key,
    ensure_vector_store_directory,
    GoogleDriveSync,
    ingest_documents,
    check_vector_store,
    ask,
    VECTOR_STORE_DIR,
    DATA_DIR,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    GOOGLE_DRIVE_FOLDER_ID,
    MEMORY
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global state
sync_status = {
    "is_syncing": False,
    "last_sync": None,
    "error": None,
    "documents_processed": 0
}

def perform_sync():
    """Perform document sync with enhanced error handling"""
    global sync_status
    try:
        sync_status.update({
            "is_syncing": True,
            "error": None,
            "documents_processed": 0
        })
        logger.info("Starting document sync...")

        # Initialize Google Drive sync
        drive_service = GoogleDriveSync(GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_DRIVE_FOLDER_ID)
        documents = drive_service.sync_drive_folder()
        logger.info(f"Found {len(documents)} documents to process")

        # Process documents
        ingest_documents(documents)

        sync_status.update({
            "last_sync": time.time(),
            "documents_processed": len(documents),
            "is_syncing": False
        })
        logger.info("Sync completed successfully")

    except Exception as e:
        logger.error(f"Sync failed: {str(e)}", exc_info=True)
        sync_status.update({
            "error": str(e),
            "is_syncing": False
        })

def start_sync_thread():
    """Start sync in a background thread if not already running"""
    if sync_status["is_syncing"]:
        return False
    
    thread = threading.Thread(target=perform_sync)
    thread.daemon = True
    thread.start()
    return True

# Initialize sync on startup
with app.app_context():
    start_sync_thread()

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle questions from Custom GPT"""
    if sync_status["is_syncing"]:
        return jsonify({
            "error": "System is currently syncing documents",
            "status": "syncing"
        }), 503
        
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
            
        answer = ask(question)
        
        return jsonify({
            "answer": answer,
            "status": "success",
            "last_sync": sync_status["last_sync"]
        })
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Check system status"""
    return jsonify({
        "is_syncing": sync_status["is_syncing"],
        "last_sync": sync_status["last_sync"],
        "error": sync_status["error"],
        "documents_processed": sync_status["documents_processed"],
        "vector_store_ready": check_vector_store()
    })

@app.route('/sync', methods=['POST'])  # 1. Correct path and method
def trigger_sync():
    """Manually trigger document synchronization"""
    if not start_sync_thread():
        return jsonify({
            "status": "already_running",
            "message": "Sync is already in progress",
            "last_sync": sync_status["last_sync"]
        }), 200

    return jsonify({
        "status": "started",
        "message": "Document sync started in background",
        "is_syncing": True
    }), 202
    
@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset conversation history"""
    try:
        global MEMORY
        MEMORY = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        return jsonify({"status": "success", "message": "Conversation history reset"})
    except Exception as e:
        logger.error(f"Failed to reset conversation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Validate environment
    check_openai_api_key()
    
    # Ensure vector store directory exists
    if not ensure_vector_store_directory():
        logger.error("Cannot create or access vector store directory")
        sys.exit(1)
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=8000) 