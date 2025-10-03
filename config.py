import os

# --- PATHS ---
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the folder containing admin-provided documents
ADMIN_DOCS_PATH = os.path.join(BASE_DIR, "admin_docs")

# Path to the folder where user-uploaded files will be stored
USER_UPLOADS_PATH = os.path.join(BASE_DIR, "user_uploads")

# Path to the folder where the persistent FAISS vector store will be saved
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

# Path to the SQLite database file
DATABASE_PATH = os.path.join(BASE_DIR, "chatbot.db")


# --- MODELS ---
# Name of the sentence-transformer model for embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Name of the Groq model for language generation
LLM_MODEL_NAME = "openai/gpt-oss-120b"


# --- RETRIEVER SETTINGS ---
# Default number of documents to retrieve
K_RETRIEVER = 5

# Default search strategy ('similarity', 'mmr', 'hybrid')
DEFAULT_SEARCH_STRATEGY = "hybrid"

# Default chunking strategy ('recursive', 'fixed_size', 'semantic')
DEFAULT_CHUNKING_STRATEGY = "recursive"


# --- CHUNKING PARAMETERS ---
# For RecursiveCharacterTextSplitter
RECURSIVE_CHUNK_SIZE = 1000
RECURSIVE_CHUNK_OVERLAP = 150

# For CharacterTextSplitter (fixed_size)
FIXED_CHUNK_SIZE = 800
FIXED_CHUNK_OVERLAP = 100

# For semantic chunking (number of clusters)
SEMANTIC_N_CLUSTERS = 10