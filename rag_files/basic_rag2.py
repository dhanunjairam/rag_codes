import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
nltk.download('punkt')

# Step 1: Load the PDF and extract text
def load_pdf(path):
    with open(path, "rb") as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()  # Using extract_text() to get the text from a page
    return text

# Step 2: Chunk text into small pieces
def smart_chunk_text(text, chunk_size=500):
    paragraphs = re.split(r'\n\s*\n', text)  # Split by empty lines (paragraphs)
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

# Step 3: Create embeddings and FAISS index
def create_faiss_index(chunks, model_name='multi-qa-MiniLM-L6-cos-v1'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return model, index, chunks

# Step 4: Retrieval function
def retrieve(query, model, index, chunks, k=10, top_n=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    # Calculate distances and sort
    scored_chunks = [(chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    scored_chunks.sort(key=lambda x: x[1])  # Sort by distance (lower is better)

    results = [chunk for chunk, _ in scored_chunks[:top_n]]
    return results


# --- MAIN ---

# Load PDF
pdf_text = load_pdf("/Users/ABC/basic_rag/laws-of-cricket-2017-code-3rd-edition-2022_1.pdf")  # <<<<<<<< put your PDF filename here

# Chunk it
chunks = smart_chunk_text(pdf_text)

# Create FAISS index
model, index, chunks = create_faiss_index(chunks)

# Test retrieval
query = "what are Fitness for play?"
top_chunks = retrieve(query, model, index, chunks)

print("\nQuery:", query)
print("\nTop relevant chunks:")
for chunk in top_chunks:
    print("\n--- Chunk ---\n", chunk)
