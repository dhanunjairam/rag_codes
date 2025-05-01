# RAG (Retrieval-Augmented Generation) System

This project implements a Retrieval-Augmented Generation (RAG) system that enhances Large Language Model (LLM) responses with relevant information retrieved from a knowledge base. The implementation demonstrates RAG at three different levels of complexity.

## Overview

RAG systems combine the power of retrieval-based and generation-based approaches to provide more accurate, contextually relevant, and factual responses. This project includes:

1. A basic RAG implementation with predefined text snippets
2. An intermediate RAG system that processes PDF documents
3. An advanced RAG system using vector databases and LLM integration

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
Create a .env file with your LLM API key (for the advanced implementation):
LLM_API_KEY = "your-api-key-here"
Project Structure
Basic_rag/
├── .env                  # Environment variables (API keys)
├── .gitignore            # Git ignore file
├── README.md             # This file
├── requirements.txt      # Project dependencies
├── data/                 # Data directory
│   └── laws-of-cricket-2017-code-3rd-edition-2022_1.pdf  # Sample PDF
└── rag_files/            # RAG implementations
    ├── basic_rag1.py     # Basic RAG with predefined text
    ├── basic_rag2.py     # Intermediate RAG with PDF processing
    └── rag_vectordb.py   # Advanced RAG with vector DB and LLM
Usage
Basic RAG (basic_rag1.py)
This implementation demonstrates the core RAG concept using predefined text snippets and FAISS for similarity search:

python rag_files/basic_rag1.py
Features:

Uses sentence-transformers for embedding generation
Implements FAISS for efficient similarity search
Works with a small set of predefined text snippets
Intermediate RAG (basic_rag2.py)
This implementation adds PDF processing and smart text chunking:

python rag_files/basic_rag2.py
Features:

Loads and processes PDF documents
Implements smart chunking to break text into meaningful segments
Uses FAISS for efficient similarity search
Retrieves multiple relevant text chunks for a given query
Advanced RAG (rag_vectordb.py)
This implementation adds ChromaDB for vector storage and OpenAI integration:

python rag_files/rag_vectordb.py
Features:

Loads and processes PDF documents
Implements smart chunking to break text into meaningful segments
Uses ChromaDB for vector storage and retrieval
Integrates with OpenAI API (via OpenRouter) to generate answers based on retrieved context
Dependencies
sentence_transformers: For generating embeddings
faiss: For efficient similarity search
numpy: For numerical operations
PyPDF2: For PDF processing
nltk: For natural language processing tasks
chromadb: For vector database storage
openai: For LLM API integration
Customization
Using Different PDFs
To use a different PDF, update the file path in the respective scripts:

# In basic_rag2.py
pdf_text = load_pdf("path/to/your/pdf/file.pdf")

# In rag_vectordb.py
pdf_text = load_pdf("path/to/your/pdf/file.pdf")
Using Different Embedding Models
To use a different embedding model, update the model name in the respective functions:

# In basic_rag1.py
model = SentenceTransformer('your-preferred-model')

# In basic_rag2.py
def create_faiss_index(chunks, model_name='your-preferred-model'):

# In rag_vectordb.py
def create_chromadb_collection(chunks, model_name="your-preferred-model"):
Using Different LLM Providers
The advanced implementation uses OpenRouter to access various LLMs. To use a different provider or model, update the client configuration in rag_vectordb.py:

client = openai.OpenAI(
  base_url="your-provider-url",
  api_key=API_KEY,
)

completion = client.chat.completions.create(
  model="your-preferred-model",
  messages=[...]
)
License
MIT
