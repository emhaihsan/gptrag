import psycopg2
import os
from psycopg2.extras import Json
from datetime import datetime

class VectorStore:
    def __init__(self):
        # Initialize database connection using environment variables
        self.connection = psycopg2.connect(dbname='rag-1234',
                                           user=os.getenv('PG_USER'),
                                           password=os.getenv('PG_PASSWORD'),
                                           host=os.getenv('PG_HOST'),
                                           port=os.getenv('PG_PORT'))
        self.cursor = self.connection.cursor()

    def create_tables(self):
        # Create necessary extensions and tables if they don't exist
        self.cursor.execute(""" CREATE EXTENSION IF NOT EXISTS vector; """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id SERIAL PRIMARY KEY,
                embedding VECTOR(1536),
                content TEXT
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id SERIAL PRIMARY KEY,
                history JSONB
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                chat_session_id INT REFERENCES chat_sessions(session_id),
                chat TEXT,
                ai_answer TEXT,
                chat_embedding VECTOR(1536),
                ai_answer_embedding VECTOR(1536)
            );

            CREATE TABLE IF NOT EXISTS token_counter (
                id SERIAL PRIMARY KEY,
                token_type VARCHAR(50),
                tokens_used INT,
                timestamp TIMESTAMP
            );
        """)
        self.connection.commit()

    def store_embedding(self, embedding, content, tokens_used):
        # Store embedding and content in the knowledge table
        embedding_str = str(embedding)
        self.cursor.execute("""
            INSERT INTO knowledge (embedding, content)
            VALUES (%s, %s)
        """, (embedding_str, content))
        self.connection.commit()
        self.store_token_count('embedding_input', tokens_used)

    def query_similar(self, embedding, limit=1):
        # Query similar embeddings from the knowledge table
        embedding_str = str(embedding)
        self.cursor.execute("""
            SELECT content FROM knowledge
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (embedding_str, limit))
        results = self.cursor.fetchall()
        return results
    
    def store_session(self, history):
        # Store chat session history and return the session ID
        self.cursor.execute("""
            INSERT INTO chat_sessions (history)
            VALUES (%s) RETURNING session_id;
        """, [Json(history)])
        session_id = self.cursor.fetchone()[0]
        self.connection.commit()
        return session_id

    def get_session(self, session_id):
        # Retrieve chat session history by session ID
        self.cursor.execute("""
            SELECT history FROM chat_sessions
            WHERE session_id = %s;
        """, (session_id,))
        history = self.cursor.fetchone()[0]
        return history
    
    def update_session(self, session_id, history):
        # Update chat session history
        self.cursor.execute("""
            UPDATE chat_sessions
            SET history = %s
            WHERE session_id = %s;
        """, (Json(history), session_id))
        self.connection.commit()

    def store_chat_history(self, session_id, chat, ai_answer, chat_embedding, ai_answer_embedding):
        # Store chat history with embeddings
        self.cursor.execute("""
            INSERT INTO chat_history (chat_session_id, chat, ai_answer, chat_embedding, ai_answer_embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, chat, ai_answer, str(chat_embedding), str(ai_answer_embedding)))
        self.connection.commit()

    def query_chat_history(self, session_id, embedding, limit=1):
        # Query similar chat history based on embedding
        embedding_str = str(embedding)
        self.cursor.execute("""
            SELECT chat, ai_answer FROM chat_history
            WHERE chat_session_id = %s
            ORDER BY chat_embedding <-> %s
            LIMIT %s;
        """, (session_id, embedding_str, limit))
        results = self.cursor.fetchall()
        return results

    def store_token_count(self, token_type, tokens_used):
        # Store token usage count
        self.cursor.execute("""
            INSERT INTO token_counter (token_type, tokens_used, timestamp)
            VALUES (%s, %s, %s)
        """, (token_type, tokens_used, datetime.now()))
        self.connection.commit()

    def query_token_usage(self, token_type, start_date, end_date):
        # Query total token usage for a specific type and date range
        self.cursor.execute("""
            SELECT SUM(tokens_used) FROM token_counter
            WHERE token_type = %s AND timestamp BETWEEN %s AND %s;
        """, (token_type, start_date, end_date))
        total_tokens_used = self.cursor.fetchone()[0]
        return total_tokens_used or 0
    
    def clear_all_data(self):
        try:
            self.cursor.execute("""
                TRUNCATE TABLE knowledge, chat_sessions, chat_history, token_counter RESTART IDENTITY CASCADE;
            """)
            self.connection.commit()
            print("Semua data berhasil dihapus.")
        except Exception as e:
            self.connection.rollback()
            print(f"Terjadi kesalahan saat menghapus data: {e}")