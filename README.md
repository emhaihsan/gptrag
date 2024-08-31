# RAG-based Chatbot with Knowledge Management

This project implements a Retrieval-Augmented Generation (RAG) based chatbot with knowledge management capabilities. It uses FastAPI for the backend, OpenAI's GPT model for text generation, and PostgreSQL with pgvector for efficient vector storage and similarity search.

## Features

- Upload and process knowledge documents (PDF and plain text)
- Embed and store knowledge chunks
- Create and manage chat sessions
- Retrieve relevant knowledge and chat history for context-aware responses
- Track token usage for different operations

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- OpenAI API key

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Install required packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up the PostgreSQL database:

   - Ensure PostgreSQL is installed and running
   - Run the `init_db.py` script to create the database and set up the vector extension:
     ```
     python init_db.py
     ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPEN_AI_API_KEY=your_openai_api_key
   PG_USER=your_postgres_user
   PG_PASSWORD=your_postgres_password
   PG_HOST=your_postgres_host
   PG_PORT=your_postgres_port
   CHATBOT_PREPROMPT=your_chatbot_system_message
   CHUNK_SIZE=200
   OVERLAP_SIZE=20
   TOP_K=5
   TOP_K_HISTORY=3
   ```

## Usage

1. Start the FastAPI server:

   ```
   uvicorn main:app --reload
   ```

2. Use the following endpoints:
   - POST `/upload-knowledge/`: Upload a PDF or text file to add to the knowledge base
   - POST `/newchat/`: Start a new chat session
   - POST `/chat/`: Send a message in a chat session
   - GET `/token_usage/`: Retrieve token usage statistics

## Code Structure

- `main.py`: FastAPI application and main logic
- `vector_store.py`: VectorStore class for database operations
- `init_db.py`: Database initialization script

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
