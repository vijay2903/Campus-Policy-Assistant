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

[User] <--> [Streamlit UI (app.py)]
|
|--- [DatabaseManager (database.py)] <--> [SQLite DB]
|
|--- [VectorStoreManager (vectorstore.py)] <--> [FAISS Index]
|
|--- [LLMManager (llm.py)] <--> [Groq API]

---

## Local Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

-   Python 3.10 or newer
-   A Groq API Key

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd Campus
    ```

2.  **Create and Activate a Virtual Environment**
    This project uses a virtual environment to manage dependencies. If your environment is named `chat`:
    ```bash
    # Create the virtual environment
    python -m venv chat

    # Activate it (Windows)
    .\chat\Scripts\activate

    # Activate it (macOS/Linux)
    source chat/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a file named `.env` in the root directory. Add your Groq API key to this file:
    ```
    GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
    ```

5.  **Add Admin Documents**
    Place the official campus PDF documents (e.g., policy handbooks, academic calendars) inside the `admin_docs/` folder. The app will process these on its first run to build the main vector store.

6.  **Run the Application**
    ```bash
    streamlit run app.py
    ```
    Your browser should open to `http://localhost:8501` with the running application.

## Usage

1.  **Sign Up / Login**: Create a new account or log in with existing credentials.
2.  **Create a Chat**: Start a new chat from the sidebar.
3.  **Configure Strategies**: Use the dropdowns in the sidebar to select your desired chunking and search strategies for the session.
4.  **Upload Documents**: Optionally, upload your own PDFs for the chatbot to consider during the current chat session.
5.  **Ask Questions**: Start chatting!

## Running Tests

To ensure the database logic is working correctly, you can run the provided unit tests:

```bash
python -m unittest discover tests
