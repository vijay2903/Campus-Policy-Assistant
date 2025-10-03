# Campus Chatbot: A RAG-Powered Assistant

A sophisticated, RAG (Retrieval-Augmented Generation)-based chatbot designed to serve as an intelligent assistant for a college campus. Users can ask questions about campus policies, procedures, and events, and receive accurate, context-aware answers sourced from official documents.

This project is built with a modular, object-oriented architecture and demonstrates a full development lifecycle, from setup and testing to a configurable and user-friendly interface.


---

## Key Features

-   **Secure User Authentication**: A complete login/signup system to manage user access and maintain separate chat histories.
-   **Persistent & Private Chats**: Users can create multiple conversations, access them later, and delete them. Each chat history is stored securely in a database.
-   **Hybrid Document Retrieval**: The RAG system can retrieve information from two sources simultaneously:
    -   A permanent **Admin Vector Store** containing official campus documents.
    -   A temporary **User Vector Store** created on-the-fly from documents uploaded by the user within a specific chat session.
-   **Advanced RAG Pipeline**:
    -   **Configurable Chunking**: Select from `recursive`, `fixed_size`, or `semantic` chunking strategies to optimize document processing.
    -   **Configurable Search**: Switch between `hybrid` (BM25 keyword + vector similarity), `similarity`, or `mmr` (Maximal Marginal Relevance) search to improve retrieval quality.
-   **Source Citations**: The chatbot's responses include citations, pointing to the source documents used to generate the answer, ensuring transparency and trust.
-   **Robust Backend**: Built with a modular, multi-file, Object-Oriented structure for maintainability and scalability.
-   **Unit Tested**: Includes a test suite for the database module to ensure data integrity and reliability.

## Tech Stack

-   **Backend**: Python
-   **Web Framework**: Streamlit
-   **LLM & RAG Orchestration**: LangChain
-   **Language Model (LLM)**: Groq (via `openai/gpt-oss-120b`)
-   **Embeddings**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`)
-   **Vector Store**: FAISS (Facebook AI Similarity Search)
-   **Database**: SQLite3
-   **UI Components**: `streamlit-chat`

## Architecture Overview

The application is designed with a clear separation of concerns, making it easy to understand, test, and extend.
