import os
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from vector_store import VectorStore
import PyPDF2
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

app = FastAPI()

vector_store = VectorStore()
vector_store.create_tables()

OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY')
CHATBOT_PREPROMPT = os.getenv('CHATBOT_PREPROMPT')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 200))
OVERLAP_SIZE = int(os.getenv('OVERLAP_SIZE', 20))
TOP_K = int(os.getenv('TOP_K', 5))
TOP_K_HISTORY = int(os.getenv('TOP_K_HISTORY', 3))



def extract_text_from_pdf(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    response_data = response.json()
    if 'error' in response_data:
        raise Exception(f"OpenAI API error: {response_data['error']}")
    tokens_used = response_data['usage']['total_tokens']
    vector_store.store_token_count('embedding', tokens_used)
    return response_data['data'][0]['embedding']

def split_text_into_chunks(text, chunk_size, overlap_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def chat_with_openai(messages):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": messages
    }
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    prompt_tokens = response_data['usage']['prompt_tokens']
    completion_tokens = response_data['usage']['completion_tokens']
    vector_store.store_token_count('completion', completion_tokens)
    vector_store.store_token_count('prompt', prompt_tokens)
    return response_data['choices'][0]['message']['content']



@app.post("/upload-knowledge/")
async def upload_knowledge(file: UploadFile = File(...)):
    print(OPENAI_API_KEY)
    try:
        if file.content_type not in ["application/pdf", "text/plain"]:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        content = await file.read()
        logging.debug(f"File content type: {file.content_type}")
        logging.debug(f"File size: {len(content)} bytes")
        
        if file.content_type == "application/pdf":
            with open("temp.pdf", "wb") as temp_file:
                temp_file.write(content)
            text = extract_text_from_pdf("temp.pdf")
            os.remove("temp.pdf")
        else:
            text = content.decode("utf-8")
        
        logging.debug(f"Extracted text length: {len(text)} characters")
        
        chunks = split_text_into_chunks(text, CHUNK_SIZE, OVERLAP_SIZE)
        logging.debug(f"Number of chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks):
            logging.debug(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                embedding = get_embedding(chunk)
                tokens_used = len(chunk.split())
                vector_store.store_embedding(embedding, chunk, tokens_used)
            except Exception as e:
                logging.error(f"Error processing chunk {i+1}: {str(e)}")
                raise
        
        return {"message": "Knowledge uploaded and split into chunks successfully"}
    except Exception as e:
        logging.exception(f"Error in upload_knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/newchat/")
async def newchat():
    messages = [
        {"role": "system", "content": CHATBOT_PREPROMPT},
    ]
    session_id = vector_store.store_session(messages)
    return {"session_id": session_id}

@app.post("/chat/")
async def chat_with_session(session_id: int, text: str):
    session_history = vector_store.get_session(session_id)
    chat_embedding = get_embedding(text)

    # Retrieve relevant knowledge
    knowledge = vector_store.query_similar(chat_embedding, limit=TOP_K)
    knowledge_texts = [item[0] for item in knowledge]

    # Retreive top-k relevant chat hisotry within the session
    previous_chats = vector_store.query_chat_history(session_id, chat_embedding, limit=TOP_K_HISTORY)
    previous_chats_texts = [f"User: {item[0]}\nChatbot: {item[1]}\n" for item in previous_chats]

    # Combine knowledge and previous chat history
    combined_context = "\n".join(previous_chats_texts + knowledge_texts)

    if combined_context:
        session_history.append({"role": "system", "content": combined_context})

    session_history.append({"role": "user", "content": text})
    response_content = chat_with_openai(session_history)
    ai_answer_embedding = get_embedding(response_content)
    vector_store.store_chat_history(session_id, text, response_content, chat_embedding, ai_answer_embedding)

    return JSONResponse(content={"response": response_content, "knowledge": knowledge_texts, "previous_chats": previous_chats_texts})

@app.get("/token_usage/")
async def token_usage(
    token_type: str,
    start_date: datetime = Query(..., description="Start date in the format YYYY-MM-DDTHH:MM:SS"),
    end_date: datetime = Query(..., description="End date in the format YYYY-MM-DDTHH:MM:SS"),
):
    if token_type not in ['embedding_input', 'completion', 'prompt']:
        raise HTTPException(status_code=400, detail="Invalid token type")
    tokens_used = vector_store.query_token_usage(token_type, start_date, end_date)
    return {"token_type": token_type, "tokens_used": tokens_used, "start_date": start_date, "end_date": end_date}

@app.post("/clear-data/")
async def clear_all_data():
    try:
        vector_store.clear_all_data()
        return {"message": "All data has been deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))