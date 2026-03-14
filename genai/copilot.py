import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

# Load everything once
_model = SentenceTransformer("all-MiniLM-L6-v2")
_index = faiss.read_index("genai/vectorstore/index.faiss")
with open("genai/vectorstore/documents.pkl", "rb") as f:
    _documents = pickle.load(f)

# _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question: str) -> str:
    # Step 1 — Embed question
    q_embedding = _model.encode([question]).astype("float32")

    # Step 2 — Search FAISS
    _, indices = _index.search(q_embedding, k=6)
    relevant_chunks = [_documents[i] for i in indices[0]]

    # Step 3 — Build context
    context = "\n\n".join([
        f"[Source: {doc['source']}]\n{doc['text']}"
        for doc in relevant_chunks
    ])

    # Step 4 — Ask Gemini
    prompt = f"""You are an expert vehicle insurance analytics assistant.
Answer ONLY using the data chunks below.
Always cite specific numbers. Never guess.
If answer not in data, say "I don't have enough data to answer that precisely."
Keep answers to 2-4 sentences.

Data Chunks:
{context}

Question: {question}

Answer:"""

    # response = _client.models.generate_content(
    #     model="gemini-1.5-flash-8b",
    #     contents=prompt
    # )
    # return response.text
    response = _client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content