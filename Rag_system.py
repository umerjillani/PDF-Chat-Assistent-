# Add this at the very top of the file, before any other imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -*- coding: utf-8 -*-
"""
Enhanced RAG system with OpenAI models, improved OCR handling, and conversational memory
For local execution (not Colab)
"""

import os
import sys
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
import argparse
import io
import re
from PIL import Image
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from datetime import datetime
import json
from dotenv import load_dotenv
import pickle
import faiss

# Load environment variables
load_dotenv()

# Google Drive imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import googleapiclient.errors

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

# CONFIGURATION SECTION
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
OCR_ENABLED = os.getenv("OCR_ENABLED", "True").lower() == "true"
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng")
MAX_CONCURRENT_OCR = int(os.getenv("MAX_CONCURRENT_OCR", "4"))
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Directory structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
VECTOR_STORE_DIR = os.path.join(CURRENT_DIR, "faiss_index")
SYNC_STATE_FILE = os.path.join(CURRENT_DIR, "sync_state.json")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Create global memory instance
MEMORY = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

class GoogleDriveSync:
    def __init__(self, service_account_file, folder_id):
        self.folder_id = folder_id
        self.service_account_file = service_account_file
        self.drive_service = None
        self.sync_state = self._load_sync_state()
        
    def _load_sync_state(self):
        if os.path.exists(SYNC_STATE_FILE):
            with open(SYNC_STATE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_sync_state(self):
        with open(SYNC_STATE_FILE, 'w') as f:
            json.dump(self.sync_state, f)
    
    def initialize_drive_service(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_file,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            self.drive_service = build('drive', 'v3', credentials=credentials)
            logger.info("Google Drive service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise
    
    def get_file_hash(self, file_path):
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def download_file(self, file_id, file_name):
        try:
            request = self.drive_service.files().get_media(fileId=file_id)
            file_path = os.path.join(DATA_DIR, file_name)
            
            with open(file_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            
            return file_path
        except Exception as e:
            logger.error(f"Error downloading file {file_name}: {e}")
            return None
    
    def sync_drive_folder(self):
        if not self.drive_service:
            self.initialize_drive_service()
        
        try:
            # Get all PDF files in the specified folder
            results = self.drive_service.files().list(
                q=f"'{self.folder_id}' in parents and mimeType='application/pdf'",
                fields="files(id, name, modifiedTime)"
            ).execute()
            
            files = results.get('files', [])
            current_files = {}
            
            for file in files:
                file_id = file['id']
                file_name = file['name']
                modified_time = file['modifiedTime']
                
                # Download file if new or modified
                if (file_id not in self.sync_state or 
                    self.sync_state[file_id]['modified_time'] != modified_time):
                    
                    file_path = self.download_file(file_id, file_name)
                    if file_path:
                        file_hash = self.get_file_hash(file_path)
                        current_files[file_id] = {
                            'name': file_name,
                            'path': file_path,
                            'modified_time': modified_time,
                            'hash': file_hash
                        }
                        logger.info(f"Downloaded/Updated: {file_name}")
                else:
                    current_files[file_id] = self.sync_state[file_id]
            
            # Remove files that no longer exist in Drive
            removed_files = set(self.sync_state.keys()) - set(current_files.keys())
            for file_id in removed_files:
                old_file_path = self.sync_state[file_id]['path']
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)
                logger.info(f"Removed: {self.sync_state[file_id]['name']}")
            
            # Update sync state
            self.sync_state = current_files
            self._save_sync_state()
            
            return list(current_files.values())
            
        except Exception as e:
            logger.error(f"Error syncing Drive folder: {e}")
            raise

def check_openai_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return api_key

def check_ocr_dependencies():
    """Check OCR dependencies and try alternative methods if needed"""
    ocr_methods = []
    
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        ocr_methods.append('pytesseract')
        logger.info("Found working pytesseract OCR")
    except Exception as e:
        logger.warning(f"Pytesseract OCR not available: {e}")
    
    try:
        import easyocr
        ocr_methods.append('easyocr')
        logger.info("Found working easyocr")
    except ImportError:
        logger.warning("EasyOCR not available. Consider installing with 'pip install easyocr'")
    
    # Check if poppler is accessible
    poppler_available = False
    try:
        from pdf2image import convert_from_path
        with tempfile.NamedTemporaryFile(suffix='.pdf') as f:
            # Create a minimal PDF
            f.write(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 22 >>\nstream\nBT /F1 12 Tf (Test) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000198 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n270\n%%EOF\n")
            f.flush()
            try:
                # Try to convert first page
                image = convert_from_path(f.name, first_page=1, last_page=1)
                if image:
                    poppler_available = True
                    logger.info("Poppler is working properly")
            except Exception as e:
                logger.warning(f"Poppler test failed: {e}")
    except ImportError:
        logger.warning("pdf2image not installed. Consider installing with 'pip install pdf2image'")
    
    if not poppler_available:
        logger.warning("""
        Poppler is not working properly. This is needed for pdf2image.
        """)
    
    return {
        'ocr_methods': ocr_methods,
        'poppler_available': poppler_available
    }

def get_pages_with_images(pdf_path):
    """Detect pages containing images using PyMuPDF"""
    doc = fitz.open(pdf_path)
    image_pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        if page.get_images():
            image_pages.append(page_num)
    
    doc.close()
    return image_pages

def extract_images_pymupdf(pdf_path, page_num):
    """Extract images from a PDF page using PyMuPDF"""
    images = []
    doc = fitz.open(pdf_path)
    
    try:
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    except Exception as e:
        logger.error(f"Error extracting images from page {page_num+1}: {str(e)}")
    finally:
        doc.close()
    
    return images

def extract_page_as_image(pdf_path, page_num):
    """Extract page as image using PyMuPDF"""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        return image
    except Exception as e:
        logger.error(f"Error converting page {page_num+1} to image: {str(e)}")
        return None
    finally:
        doc.close()

def perform_ocr_on_page(pdf_path, page_num, ocr_methods):
    """Perform OCR on a single page using best available method"""
    logger.info(f"Processing OCR for page {page_num+1}")
    result_text = ""
    
    # First try to extract the page as an image using PyMuPDF
    page_image = extract_page_as_image(pdf_path, page_num)
    
    if page_image:
        if 'pytesseract' in ocr_methods:
            try:
                import pytesseract
                result_text = pytesseract.image_to_string(page_image, lang=TESSERACT_LANG)
                logger.info(f"Used pytesseract OCR on page {page_num+1}")
            except Exception as e:
                logger.warning(f"Pytesseract OCR failed on page {page_num+1}: {str(e)}")
        
        if not result_text.strip() and 'easyocr' in ocr_methods:
            try:
                import easyocr
                reader = easyocr.Reader(['en'])  # Initialize reader with English
                results = reader.readtext(np.array(page_image))
                result_text = "\n".join([text for _, text, _ in results])
                logger.info(f"Used EasyOCR on page {page_num+1}")
            except Exception as e:
                logger.warning(f"EasyOCR failed on page {page_num+1}: {str(e)}")
    
    if result_text.strip():
        return {
            "page_content": result_text,
            "metadata": {
                "source": pdf_path,
                "page": page_num,
                "is_ocr": True,
                "contains_images": True
            }
        }
    else:
        logger.warning(f"No text extracted from page {page_num+1}")
        return None

def extract_image_text_concurrent(pdf_path, page_nums, ocr_methods):
    """Extract text from multiple pages using OCR with concurrency"""
    extracted = []
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_OCR) as executor:
        future_to_page = {
            executor.submit(perform_ocr_on_page, pdf_path, page_num, ocr_methods): page_num 
            for page_num in page_nums
        }
        
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                result = future.result()
                if result:
                    extracted.append(result)
            except Exception as e:
                logger.error(f"Page {page_num+1} OCR failed with error: {str(e)}")
    
    return extracted

def get_pdf_text_from_pymupdf(pdf_path):
    """Extract text directly from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text()
            
            # Only add pages with actual text content
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "is_ocr": False
                    }
                ))
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num+1}: {str(e)}")
    
    doc.close()
    return pages

def get_document_hash(file_path):
    """Calculate hash of a document for change detection"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None

def get_processed_documents():
    """Get list of previously processed documents"""
    processed_file = os.path.join(VECTOR_STORE_DIR, "processed_documents.json")
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading processed documents file: {e}")
    return {}

def save_processed_documents(processed_docs):
    """Save list of processed documents"""
    processed_file = os.path.join(VECTOR_STORE_DIR, "processed_documents.json")
    try:
        with open(processed_file, 'w') as f:
            json.dump(processed_docs, f)
    except Exception as e:
        logger.error(f"Error saving processed documents file: {e}")

def remove_document_from_vector_store(document_path):
    """Remove a document from the vector store"""
    try:
        # Load the existing vector store
        embedding = OpenAIEmbeddings()
        vector_store = load_vector_store(VECTOR_STORE_DIR, embedding)
        
        if vector_store is None:
            logger.warning("No vector store found to remove document from.")
            return False
            
        # Get all documents with the matching source
        results = vector_store.similarity_search(document_path, k=1)
        
        if results:
            # Delete the documents
            vector_store.delete([doc.metadata.get('document_id') for doc in results])
            # Save the updated vector store
            save_vector_store(vector_store, VECTOR_STORE_DIR)
            logger.info(f"Removed document {document_path} from vector store")
            return True
    except Exception as e:
        logger.error(f"Error removing document from vector store: {e}")
    return False

def ensure_vector_store_directory():
    """Ensure vector store directory exists and has proper permissions"""
    try:
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(VECTOR_STORE_DIR, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Error ensuring vector store directory: {e}")
        return False

def save_vector_store(vector_store, index_path):
    """Save FAISS vector store to disk"""
    try:
        # Save the FAISS index
        faiss.write_index(vector_store.index, os.path.join(index_path, "faiss.index"))
        
        # Save the document store
        with open(os.path.join(index_path, "docstore.pkl"), "wb") as f:
            pickle.dump(vector_store.docstore, f)
            
        # Save the index to docstore mapping
        with open(os.path.join(index_path, "index_to_docstore_id.pkl"), "wb") as f:
            pickle.dump(vector_store.index_to_docstore_id, f)
            
        logger.info("Vector store saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving vector store: {e}")
        return False

def load_vector_store(index_path, embeddings):
    """Load FAISS vector store from disk"""
    try:
        if not os.path.exists(os.path.join(index_path, "faiss.index")):
            return None
            
        # Load the FAISS index
        index = faiss.read_index(os.path.join(index_path, "faiss.index"))
        
        # Load the document store
        with open(os.path.join(index_path, "docstore.pkl"), "rb") as f:
            docstore = pickle.load(f)
            
        # Load the index to docstore mapping
        with open(os.path.join(index_path, "index_to_docstore_id.pkl"), "rb") as f:
            index_to_docstore_id = pickle.load(f)
            
        # Create and return the vector store with proper embedding object
        return FAISS(embeddings, index, docstore, index_to_docstore_id)
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

def ingest_documents(documents):
    logger.info("Processing documents...")
    try:
        # Get previously processed documents
        processed_docs = get_processed_documents()
        current_docs = {}
        all_chunks = []
        
        # Log total number of documents to process
        logger.info(f"Total documents to process: {len(documents)}")
        
        for doc_info in documents:
            pdf_path = doc_info['path']
            logger.info(f"Starting processing of document: {os.path.basename(pdf_path)}")
            
            current_hash = get_document_hash(pdf_path)
            if not current_hash:
                logger.warning(f"Skipping {pdf_path} due to hash calculation error")
                continue
                
            current_docs[pdf_path] = {
                'hash': current_hash,
                'modified_time': doc_info.get('modified_time', '')
            }
            
            # Check if document needs processing
            if (pdf_path not in processed_docs or 
                processed_docs[pdf_path]['hash'] != current_hash):
                
                logger.info(f"Processing new/modified document: {os.path.basename(pdf_path)}")
                
                # Extract text using PyMuPDF
                pages = get_pdf_text_from_pymupdf(pdf_path)
                logger.info(f"Extracted {len(pages)} text pages from {os.path.basename(pdf_path)}")
                
                if OCR_ENABLED:
                    image_pages = get_pages_with_images(pdf_path)
                    if image_pages:
                        logger.info(f"Found {len(image_pages)} pages with images in {os.path.basename(pdf_path)}")
                        ocr_status = check_ocr_dependencies()
                        image_texts = extract_image_text_concurrent(pdf_path, image_pages, ocr_status['ocr_methods'])
                        pages.extend([Document(**item) for item in image_texts])
                        logger.info(f"Added {len(image_texts)} OCR pages from {os.path.basename(pdf_path)}")
                
                # Add document metadata
                for page in pages:
                    page.metadata.update({
                        'source': pdf_path,
                        'document_name': os.path.basename(pdf_path),
                        'document_id': current_hash
                    })
                
                all_chunks.extend(pages)
                logger.info(f"Total chunks after processing {os.path.basename(pdf_path)}: {len(all_chunks)}")
            else:
                logger.info(f"Skipping unchanged document: {os.path.basename(pdf_path)}")
        
        # Remove documents that no longer exist
        for old_path in processed_docs:
            if old_path not in current_docs:
                logger.info(f"Removing deleted document: {os.path.basename(old_path)}")
                remove_document_from_vector_store(old_path)
        
        if all_chunks:
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(all_chunks)
            logger.info(f"Created {len(chunks)} chunks from {len(all_chunks)} pages")
            
            # Log chunk distribution by document
            doc_chunks = {}
            for chunk in chunks:
                doc_name = chunk.metadata.get('document_name', 'unknown')
                doc_chunks[doc_name] = doc_chunks.get(doc_name, 0) + 1
            
            logger.info("Chunk distribution by document:")
            for doc_name, count in doc_chunks.items():
                logger.info(f"- {doc_name}: {count} chunks")

            # Create embeddings
            embedding = OpenAIEmbeddings()
            
            try:
                # Try to load existing vector store
                vector_store = load_vector_store(VECTOR_STORE_DIR, embedding)
                
                if vector_store is None:
                    # Create new vector store if none exists
                    # Extract texts and metadatas from chunks
                    texts = [doc.page_content for doc in chunks]
                    metadatas = [doc.metadata for doc in chunks]
                    
                    # Create vector store with proper metadata handling
                    vector_store = FAISS.from_texts(
                        texts=texts,
                        embedding=embedding,
                        metadatas=metadatas
                    )
                    logger.info("Created new vector store")
                else:
                    # Add new documents to existing store
                    texts = [doc.page_content for doc in chunks]
                    metadatas = [doc.metadata for doc in chunks]
                    vector_store.add_texts(texts=texts, metadatas=metadatas)
                    logger.info("Added documents to existing vector store")
                
                # Save the vector store
                if not save_vector_store(vector_store, VECTOR_STORE_DIR):
                    raise Exception("Failed to save vector store")
                
                logger.info("Vector store updated successfully")
                
            except Exception as e:
                logger.error(f"Error with vector store operations: {e}")
                raise
        
        # Update processed documents record
        save_processed_documents(current_docs)
        
        # Reset memory
        global MEMORY
        MEMORY = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise
    
    
def check_vector_store():
    """Check if the vector store is properly loaded and has documents"""
    try:
        embedding = OpenAIEmbeddings()
        vector_store = load_vector_store(VECTOR_STORE_DIR, embedding)
        
        if vector_store is None:
            logger.warning("Vector store does not exist!")
            return False
        
        # Test a simple query to ensure it works
        results = vector_store.similarity_search("test", k=1)
        logger.info(f"Vector store test query returned {len(results)} results")
        
        return True
    except Exception as e:
        logger.error(f"Vector store check failed: {e}")
        return False
    
def enhance_query(query):
    """Enhance the query for better retrieval"""
    # Use OpenAI to generate search queries
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        enhanced_query = llm.invoke(
            f"""Given a user question, generate an optimized search query for retrieving relevant information from a document database.
            
            Original question: "{query}"
            
            Generate a search query that:
            1. Includes key terms and concepts from the original question
            2. Adds relevant synonyms or alternative phrasings
            3. Removes filler words and focuses on important terms
            4. Is concise but comprehensive
            
            Search query:"""
        ).content
        
        # Clean up the response to get just the query
        enhanced_query = enhanced_query.strip().strip('"\'')
        
        # If the enhanced query is too different, use a combination
        if len(enhanced_query) > len(query) * 2 or len(enhanced_query) < len(query) / 2:
            return f"{query} {enhanced_query}"
        return enhanced_query
    except Exception as e:
        logger.warning(f"Query enhancement failed: {e}")
        return query  # Fall back to original query

def create_conversational_rag_chain():
    """Create a RAG chain with conversation memory"""
    logger.info("\nInitializing conversational RAG pipeline...")
    try:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE
        )

        # Load vector store
        embedding = OpenAIEmbeddings()
        vector_store = load_vector_store(VECTOR_STORE_DIR, embedding)
        
        if vector_store is None:
            raise ValueError("Vector store is empty or not properly initialized")

        # Create retriever with proper configuration
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 10,
                "lambda_mult": 0.7
            }
        )

        # Build ConversationalRetrievalChain with memory
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=MEMORY,
            return_source_documents=True,
            output_key="answer"
        )
        
        return chain
    except Exception as e:
        logger.error(f"Error initializing conversational RAG chain: {str(e)}")
        raise

def ask(query: str):
    try:
        print(f"\nQuestion: {query}")
        
        # Try to enhance the query
        enhanced_query = enhance_query(query)
        logger.debug(f"Enhanced query: {enhanced_query}")
        
        # Create the conversational RAG chain
        chain = create_conversational_rag_chain()
        
        # Execute the query with conversation memory
        result = chain.invoke({"question": enhanced_query})
        
        # Print the answer
        print("\nAnswer:", result["answer"])
        
        # Process and print sources with better formatting
        print("\nSources:")
        seen_pages = set()  # Track unique pages
        
        # Check if source_documents exists in the result
        source_docs = result.get("source_documents", [])
        for i, doc in enumerate(source_docs, 1):
            page_num = doc.metadata.get('page', 'N/A')
            if page_num != 'N/A':
                page_num += 1  # Convert to 1-indexed for display
                
            # Avoid showing duplicate page references
            page_key = f"{doc.metadata.get('source', 'Unknown')}_{page_num}"
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            
            source_type = "OCR" if doc.metadata.get("is_ocr", False) else "Text"
            print(f"{i}. Page {page_num} ({source_type}) - {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
        
        # Print conversation history for debugging if needed
        # print("\nConversation History:")
        # for message in MEMORY.chat_memory.messages:
        #     print(f"{message.type}: {message.content[:50]}...")
        
        return result["answer"]
    except Exception as e:
        print(f"\nError processing question: {str(e)}")
        return f"Error: {str(e)}"

def main():
    # Check OpenAI API key
    check_openai_api_key()
    
    print("Starting RAG application...")
    print("===========================")
    print(f"Vector store directory: {VECTOR_STORE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"OpenAI model: {OPENAI_MODEL}")
    
    # Ensure vector store directory exists before processing
    if not ensure_vector_store_directory():
        print("Error: Cannot create or access vector store directory")
        print(f"Please check permissions for: {VECTOR_STORE_DIR}")
        sys.exit(1)
    
    # Initialize Google Drive sync
    drive_service = GoogleDriveSync(GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_DRIVE_FOLDER_ID)
    
    # Get documents from Google Drive
    documents = drive_service.sync_drive_folder()
    
    # Process documents
    try:
        ingest_documents(documents)
    except Exception as e:
        print(f"\nError processing documents: {e}")
        sys.exit(1)
    
    # Check if vector store is functioning
    if not check_vector_store():
        logger.warning("Vector store appears to be empty or not functioning properly.")
        print("\nWARNING: Vector store may be empty. Document processing may have failed.")
        
        # List files in vector store directory
        if os.path.exists(VECTOR_STORE_DIR):
            print(f"Files in {VECTOR_STORE_DIR}:")
            for file in os.listdir(VECTOR_STORE_DIR):
                file_path = os.path.join(VECTOR_STORE_DIR, file)
                file_size = os.path.getsize(file_path) / 1024  # Size in KB
                print(f"- {file} ({file_size:.2f} KB)")
        else:
            print(f"Vector store directory {VECTOR_STORE_DIR} does not exist!")
        
        # Check for specific required files
        essential_files = ["faiss.index", "docstore.pkl", "index_to_docstore_id.pkl"]
        missing_files = [f for f in essential_files if not os.path.exists(os.path.join(VECTOR_STORE_DIR, f))]
        if missing_files:
            print(f"Missing essential vector store files: {missing_files}")
            
            # Try to recreate the vector store
            print("\nAttempting to recreate vector store...")
            try:
                # Clear the vector store directory
                if os.path.exists(VECTOR_STORE_DIR):
                    import shutil
                    shutil.rmtree(VECTOR_STORE_DIR)
                os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
                
                # Reprocess documents
                ingest_documents(documents)
                
                # Verify recreation
                if check_vector_store():
                    print("Vector store successfully recreated!")
                else:
                    print("Failed to recreate vector store.")
                    sys.exit(1)
            except Exception as e:
                print(f"Error recreating vector store: {e}")
                sys.exit(1)
        
        # Ask if user wants to force reprocessing
        response = input("Would you like to force reprocessing of all documents? (y/n): ")
        if response.lower() == 'y':
            print("Clearing processed documents record and reprocessing...")
            # Clear processed documents record to force reprocessing
            save_processed_documents({})
            # Try processing again
            ingest_documents(documents)
            
            # Check again
            if not check_vector_store():
                print("\nERROR: Vector store still not functioning properly after reprocessing.")
                sys.exit(1)
    
    # Interactive Question Loop
    print("\nSystem ready! Type 'exit' to quit, 'clear' to reset conversation history.")
    
    try:
        while True:
            question = input("\nEnter your question: ")
            
            if question.lower() in ('exit', 'quit'):
                break
                
            if question.lower() in ('clear', 'reset'):
                print("Clearing conversation history...")
                global MEMORY
                MEMORY = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                continue
                
            if question.lower() in ('debug', 'status'):
                print("\nSystem Status:")
                print(f"- Vector store location: {VECTOR_STORE_DIR}")
                try:
                    embedding = OpenAIEmbeddings()
                    vector_store = FAISS(embedding.embed_query, None, None, None)
                    collection = vector_store._collection
                    count = collection.count()
                    print(f"- Vector store document count: {count}")
                    print("- Vector store status: OK")
                except Exception as e:
                    print(f"- Vector store error: {e}")
                continue
                
            # Process the question
            answer = ask(question)
            
            # If answer indicates information not found, suggest reprocessing
            if "couldn't find this specific information" in answer.lower():
                print("\nTIP: If you believe this information should be in the documents,")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    
    print("Application terminated.")

if __name__ == "__main__":
    main()