import streamlit as st
from streamlit_chat import message
import os
from config import USER_UPLOADS_PATH
from database import DatabaseManager
from vectorstore import VectorStoreManager
from llm import LLMManager

class CampusChatbotApp:
    """The main application class for the Streamlit chatbot."""

    def __init__(self):
        st.set_page_config(page_title="Campus Chatbot", layout="wide")
        self.db = DatabaseManager()
        self.vsm = VectorStoreManager()
        self.llm = LLMManager()

    def _initialize_session_state(self):
        """Initializes session state variables."""
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "user_id" not in st.session_state:
            st.session_state.user_id = None
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "user_vector_store" not in st.session_state:
            st.session_state.user_vector_store = None
        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = None

    def _show_login_page(self):
        """Displays the login and signup forms."""
        st.title("Campus Chatbot Login")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    user_id = self.db.login(username, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        with col2:
            st.subheader("Sign Up")
            with st.form("signup_form"):
                new_username = st.text_input("Choose a Username")
                new_password = st.text_input("Choose a Password", type="password")
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    success, msg = self.db.signup(new_username, new_password)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    
    def _handle_file_uploads(self, chunking_strategy):
        """Handles file uploads and creates a user-specific vector store."""
        uploaded_files = st.file_uploader(
            "Upload PDF documents for this chat session", 
            type="pdf", 
            accept_multiple_files=True
        )
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                user_chat_uploads_dir = os.path.join(USER_UPLOADS_PATH, str(st.session_state.current_chat_id))
                os.makedirs(user_chat_uploads_dir, exist_ok=True)
                
                saved_file_paths = []
                for file in uploaded_files:
                    try:
                        file_path = os.path.join(user_chat_uploads_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        # Check if this file is already recorded to avoid duplicates
                        if file_path not in self.db.get_uploads_for_chat(st.session_state.current_chat_id):
                            self.db.add_upload(st.session_state.current_chat_id, file_path)
                        saved_file_paths.append(file_path)
                    except Exception as e:
                        st.error(f"Error uploading {file.name}: {e}")
            
            # Recreate vector store with all files for this chat
            all_chat_files = self.db.get_uploads_for_chat(st.session_state.current_chat_id)
            if all_chat_files:
                st.session_state.user_vector_store = self.vsm.create_user_vectorstore(
                    all_chat_files, chunking_strategy
                )
                st.success(f"{len(uploaded_files)} file(s) processed successfully!")
        
        # Ensure user_vector_store is loaded for existing files on rerun
        elif st.session_state.user_vector_store is None:
             all_chat_files = self.db.get_uploads_for_chat(st.session_state.current_chat_id)
             if all_chat_files:
                 st.session_state.user_vector_store = self.vsm.create_user_vectorstore(
                    all_chat_files, chunking_strategy
                )


    def _show_main_app(self):
        """Displays the main chat interface."""
        with st.sidebar:
            st.title("Campus Chatbot")
            st.write(f"Welcome!")

            if st.button("New Chat"):
                st.session_state.current_chat_id = None
                st.session_state.chat_history = []
                st.session_state.user_vector_store = None
                st.session_state.rag_chain = None

            st.subheader("Your Chats")
            user_chats = self.db.get_user_chats(st.session_state.user_id)
            for chat_id, chat_name in user_chats:
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    if st.button(f"{chat_name} (ID: {chat_id})", key=f"chat_{chat_id}", use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.chat_history = self.db.get_chat_history(chat_id)
                        st.session_state.user_vector_store = None # Will be lazy-loaded
                        st.session_state.rag_chain = None

                with col2:
                    if st.button("🗑️", key=f"del_{chat_id}"):
                        self.db.delete_chat(chat_id)
                        if st.session_state.current_chat_id == chat_id:
                            st.session_state.current_chat_id = None
                            st.session_state.chat_history = []
                        st.rerun()

            st.subheader("Settings")
            chunking_strategy = st.selectbox("Chunking Strategy", ["recursive", "fixed_size", "semantic"])
            search_strategy = st.selectbox("Search Strategy", ["hybrid", "similarity", "mmr"])

            if st.button("Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        st.header("Chat with Campus Policies")
        
        if st.session_state.current_chat_id is None:
            st.info("Start a new chat or select an existing one from the sidebar.")
            return

        # Handle file uploads and update retriever
        self._handle_file_uploads(chunking_strategy)
        retriever = self.vsm.get_retriever(st.session_state.user_vector_store, search_strategy)
        st.session_state.rag_chain = self.llm.get_rag_chain(retriever)

        # Display chat history
        # ---- THIS IS THE CORRECTED PART ----
        for i, msg in enumerate(st.session_state.chat_history):
            role = "user" if msg["type"] == "human" else "assistant"
            # Use the index 'i' to create a unique key for each message
            message(msg["content"], is_user=(role == "user"), key=f"msg_{st.session_state.current_chat_id}_{i}")

        # Chat input
        user_query = st.chat_input("Ask a question about campus policies or documents...")
        if user_query:
            self.db.add_message(st.session_state.current_chat_id, "human", user_query)
            
            with st.spinner("Thinking..."):
                response = self.llm.generate_response(
                    st.session_state.rag_chain,
                    st.session_state.chat_history,
                    user_query,
                    session_id=str(st.session_state.current_chat_id)
                )
            
            self.db.add_message(st.session_state.current_chat_id, "ai", response["answer"])
            st.session_state.chat_history = self.db.get_chat_history(st.session_state.current_chat_id)
            st.rerun()

    def run(self):
        """The main execution method of the app."""
        self._initialize_session_state()
        if not st.session_state.logged_in:
            self._show_login_page()
        else:
            # Create a new chat if one isn't selected
            if st.session_state.current_chat_id is None:
                st.session_state.current_chat_id = self.db.create_chat(st.session_state.user_id)
                st.session_state.chat_history = []
            self._show_main_app()

if __name__ == "__main__":
    app = CampusChatbotApp()
    app.run()