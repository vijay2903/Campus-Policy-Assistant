import os
from typing import List, Union
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever  # <-- CORRECTED IMPORT
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
import numpy as np
from config import (
    ADMIN_DOCS_PATH, VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME, K_RETRIEVER,
    DEFAULT_CHUNKING_STRATEGY, DEFAULT_SEARCH_STRATEGY, RECURSIVE_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP, SEMANTIC_N_CLUSTERS
)

class VectorStoreManager:
    """Manages vector store creation, loading, and retrieval."""

    def __init__(self):
        os.makedirs(ADMIN_DOCS_PATH, exist_ok=True)
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.admin_vectorstore_path = os.path.join(VECTOR_STORE_PATH, "admin_faiss_index")
        self.admin_vectorstore = self._load_or_create_admin_vectorstore()

    def _load_documents_from_path(self, path: str) -> List[Document]:
        """Loads PDF documents from a given directory path."""
        documents = []
        for filename in os.listdir(path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return documents

    def _semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """Splits documents based on semantic clustering."""
        print("Performing semantic chunking...")
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            print("Scikit-learn is not installed. Falling back to recursive chunking.")
            return self._recursive_chunking(documents)

        # Combine all documents into a single text block and then split into sentences
        full_text = ". ".join([doc.page_content.replace("\n", " ") for doc in documents])
        sentences = [s.strip() for s in full_text.split(". ") if s.strip()]

        if not sentences or len(sentences) < SEMANTIC_N_CLUSTERS:
            return self._recursive_chunking(documents)

        sentence_embeddings = self.embeddings.embed_documents(sentences)
        
        kmeans = KMeans(n_clusters=SEMANTIC_N_CLUSTERS, random_state=42, n_init='auto').fit(sentence_embeddings)
        
        chunks = [[] for _ in range(SEMANTIC_N_CLUSTERS)]
        for i, sentence in enumerate(sentences):
            cluster_id = kmeans.labels_[i]
            chunks[cluster_id].append(sentence)
        
        final_chunks_text = [". ".join(chunk) for chunk in chunks if chunk]
        return [Document(page_content=chunk) for chunk in final_chunks_text]

    def _recursive_chunking(self, documents: List[Document]) -> List[Document]:
        print("Performing recursive chunking...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RECURSIVE_CHUNK_SIZE, 
            chunk_overlap=RECURSIVE_CHUNK_OVERLAP
        )
        return text_splitter.split_documents(documents)

    def _fixed_size_chunking(self, documents: List[Document]) -> List[Document]:
        print("Performing fixed-size chunking...")
        text_splitter = CharacterTextSplitter(
            separator="\n", 
            chunk_size=FIXED_CHUNK_SIZE, 
            chunk_overlap=FIXED_CHUNK_OVERLAP
        )
        return text_splitter.split_documents(documents)

    def get_chunks(self, documents: List[Document], strategy: str = DEFAULT_CHUNKING_STRATEGY) -> List[Document]:
        if strategy == "semantic":
            return self._semantic_chunking(documents)
        elif strategy == "fixed_size":
            return self._fixed_size_chunking(documents)
        else:
            return self._recursive_chunking(documents)

    def _load_or_create_admin_vectorstore(self) -> FAISS:
        """Loads the admin vector store or creates it if it doesn't exist."""
        if os.path.exists(self.admin_vectorstore_path):
            print("Loading existing admin vector store...")
            return FAISS.load_local(self.admin_vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating new admin vector store...")
            admin_docs = self._load_documents_from_path(ADMIN_DOCS_PATH)
            if not admin_docs:
                print("No admin documents found. Creating an empty vector store.")
                # FAISS requires at least one document
                return FAISS.from_texts(["This is a placeholder document for an empty admin store."], self.embeddings)
            
            chunks = self.get_chunks(admin_docs, strategy=DEFAULT_CHUNKING_STRATEGY)
            vectorstore = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            vectorstore.save_local(self.admin_vectorstore_path)
            print("Admin vector store created and saved.")
            return vectorstore

    def create_user_vectorstore(self, file_paths: List[str], chunking_strategy: str) -> Union[FAISS, None]:
        """Creates an in-memory vector store for user-uploaded files."""
        if not file_paths:
            return None
        
        documents = []
        for path in file_paths:
            if os.path.exists(path) and path.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading user file {path}: {e}")

        if not documents:
            return None

        chunks = self.get_chunks(documents, strategy=chunking_strategy)
        return FAISS.from_documents(documents=chunks, embedding=self.embeddings)

    def get_retriever(self, user_vs: FAISS = None, search_type: str = DEFAULT_SEARCH_STRATEGY):
        """Creates a retriever combining admin and optional user vector stores."""
        
        # Base retriever is always the admin store
        admin_retriever = self.admin_vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVER})
        
        all_docs = list(self.admin_vectorstore.docstore._dict.values())
        if user_vs and hasattr(user_vs.docstore, '_dict'):
             all_docs.extend(list(user_vs.docstore._dict.values()))
        
        # If no user store is provided, behavior depends on search_type
        if not user_vs:
            if search_type == "hybrid" and len(all_docs) > 1:
                bm25_retriever = BM25Retriever.from_documents(all_docs)
                bm25_retriever.k = K_RETRIEVER
                return EnsembleRetriever(retrievers=[admin_retriever, bm25_retriever], weights=[0.5, 0.5])
            if search_type == "mmr":
                return self.admin_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": K_RETRIEVER})
            return admin_retriever # Default to similarity

        # --- If a user store IS provided ---
        user_retriever = user_vs.as_retriever(search_kwargs={"k": K_RETRIEVER})
        
        # For hybrid search, combine everything for BM25 and FAISS retrievers
        if search_type == "hybrid" and len(all_docs) > 1:
            # Recreate a combined FAISS store for a unified similarity search
            combined_vs = FAISS.from_documents(all_docs, self.embeddings)
            faiss_retriever = combined_vs.as_retriever(search_kwargs={"k": K_RETRIEVER})
            
            bm25_retriever = BM25Retriever.from_documents(all_docs)
            bm25_retriever.k = K_RETRIEVER
            
            return EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5])

        # For non-hybrid search with user_vs, we can use EnsembleRetriever to query both
        # This is a simple way to combine results without rebuilding a new FAISS index
        return EnsembleRetriever(retrievers=[admin_retriever, user_retriever], weights=[0.5, 0.5])