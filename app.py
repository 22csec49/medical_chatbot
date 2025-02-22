import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medicalbot"

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ Missing API keys! Check your .env file.")

app = Flask(__name__)
CORS(app)

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Verify that the index exists in Pinecone
index_list = pc.list_indexes().names()
if INDEX_NAME not in index_list:
    raise ValueError(f"❌ Index '{INDEX_NAME}' not found in Pinecone. Check your Pinecone console.")

# Initialize the retriever using the HuggingFace Embeddings model
retriever = PineconeVectorStore.from_existing_index(
    INDEX_NAME,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
print("✅ Retriever loaded successfully!")

# In-memory chat history (consider using a persistent store for production)
chat_histories = {}

def get_chat_history(session_id):
    """Retrieve chat history for a session, or create a new one."""
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def generate_response_with_retry(prompt, max_retries=3, delay=5):
    """Handles Gemini AI API rate limits with retry logic."""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except genai.RateLimitError:
            print(f"⚠️ Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt + 1})")
            time.sleep(delay)
    return "❌ Error: Rate limit exceeded. Please try again later."

def custom_rag_chain(query, session_id):
    """Retrieves relevant documents and generates an answer using chat history."""
    history = get_chat_history(session_id)
    
    # Retrieve documents relevant to the query
    retrieved_docs = retriever.as_retriever().invoke(query)
    past_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])

    if retrieved_docs:
        retrieved_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = (
            f"Chat History:\n{past_messages}\n\n"
            f"Based on the following medical references, answer the question:\n\n{retrieved_text}\n\n"
            f"Question: {query}"
        )
    else:
        print("⚠️ No relevant documents found. Answering from general knowledge...")
        prompt = f"Chat History:\n{past_messages}\n\nQuestion: {query}"

    response = generate_response_with_retry(prompt)

    # Update chat history
    history.add_user_message(query)
    history.add_ai_message(response)

    return response

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "")
    session_id = data.get("session_id", "default")  # Unique session identifier

    if not query:
        return jsonify({"error": "No message provided"}), 400

    response = custom_rag_chain(query, session_id)
    return jsonify({"response": response})

if __name__ == "__main__":
    # Bind to the port provided by the environment (default to 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
