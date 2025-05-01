from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import re
import nltk
import openai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")
nltk.data.path.append('/Users/ABC/nltk_data')
nltk.download('punkt')


# --- Step 1: Load PDF ---
def load_pdf(path):
    with open(path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# --- Step 2: Smart Chunk Text ---
def smart_chunk_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []

    for para in paragraphs:
        sentences = nltk.sent_tokenize(para)
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) <= chunk_size:
                current_chunk += " " + sentence
                current_length += len(words)
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_length = len(words)

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

# --- Step 3: Create ChromaDB collection ---
def create_chromadb_collection(chunks, model_name="all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)

    # Initialize chroma client
    chroma_client = chromadb.Client()

    # Create or get collection
    collection = chroma_client.get_or_create_collection(name="pdf_chunks")

    # Encode and add
    embeddings = model.encode(chunks)

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"doc_{i}"]
        )

   
    return model, collection

# --- Step 4: Retrieve from ChromaDB ---
def retrieve(query, model, collection, n_results=3):
    query_embedding = model.encode([query])[0]  # [0] because model.encode returns a list
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents"]
    )
    return results["documents"][0]

# --- MAIN ---

# Load PDF
pdf_text = load_pdf("laws-of-cricket-2017-code-3rd-edition-2022_1.pdf")

# Chunk it
chunks = smart_chunk_text(pdf_text)

# Create DB
model, collection = create_chromadb_collection(chunks)

# Query
query = "what if no play takes place on the first day of a match ?"
top_chunks = retrieve(query, model, collection)
context = "\n\n".join(top_chunks)
prompt = f"""You are a helpful assistant. Based on the context below, answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""


client = openai.OpenAI(

  base_url="https://openrouter.ai/api/v1",

  api_key=API_KEY,

)

completion = client.chat.completions.create(


  model="meta-llama/llama-3.1-8b-instruct",

  messages=[
        {"role": "system", "content": "You answer questions based on given context only."},
        {"role": "user", "content": prompt}
    ]

)

answer = completion.choices[0].message.content
print("\nAnswer:", answer)

# print("\nQuery:", query)
# print("\nTop relevant chunks:")
# for chunk in top_chunks:
#     print("\n--- Chunk ---\n", chunk)
