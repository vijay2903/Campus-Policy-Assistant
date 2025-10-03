import sqlite3
import hashlib
import json
from config import DATABASE_PATH

class DatabaseManager:
    """Manages all database operations for the chatbot."""

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        """Returns a database connection."""
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        """Creates database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )
            """)
            # Chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    chat_name TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (id)
                )
            """)
            # Uploads table to link files to chats
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS uploads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (id)
                )
            """)
            conn.commit()

    def _hash_password(self, password):
        """Hashes a password for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()

    def signup(self, username, password):
        """Signs up a new user."""
        if not username or not password:
            return False, "Username and password cannot be empty."
        password_hash = self._hash_password(password)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
                conn.commit()
                return True, "Signup successful."
            except sqlite3.IntegrityError:
                return False, "Username already exists."

    def login(self, username, password):
        """Logs in a user and returns user_id if successful."""
        password_hash = self._hash_password(password)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ? AND password_hash = ?", (username, password_hash))
            user = cursor.fetchone()
            return user[0] if user else None

    def create_chat(self, user_id, chat_name="New Chat"):
        """Creates a new chat session for a user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO chats (user_id, chat_name) VALUES (?, ?)", (user_id, chat_name))
            conn.commit()
            return cursor.lastrowid

    def get_user_chats(self, user_id):
        """Retrieves all chats for a given user."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, chat_name FROM chats WHERE user_id = ? ORDER BY id DESC", (user_id,))
            return cursor.fetchall()

    def get_chat_history(self, chat_id):
        """Retrieves all messages for a given chat."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
            # Convert to the format expected by LangChain's memory
            return [{"type": row[0], "content": row[1]} for row in cursor.fetchall()]

    def add_message(self, chat_id, role, content):
        """Adds a message to the chat history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)", (chat_id, role, content))
            conn.commit()

    def delete_chat(self, chat_id):
        """Deletes a chat and all its associated messages and uploads."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            cursor.execute("DELETE FROM uploads WHERE chat_id = ?", (chat_id,))
            cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()

    def add_upload(self, chat_id, file_path):
        """Records a file upload associated with a chat."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO uploads (chat_id, file_path) VALUES (?, ?)", (chat_id, file_path))
            conn.commit()
    
    def get_uploads_for_chat(self, chat_id):
        """Retrieves all file paths for a given chat."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM uploads WHERE chat_id = ?", (chat_id,))
            return [row[0] for row in cursor.fetchall()]