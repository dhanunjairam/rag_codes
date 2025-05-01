from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Prepare documents
documents = [
    "The sun is the center of the solar system.",
    "Water boils at 100 degrees Celsius.",
    "Elephants are the largest land animals.",
    "Python is a popular programming language.",
    "Mount Everest is the highest mountain on Earth."
]

# Step 2: Create Embeddings (encode smell of each doc)
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)

# Step 3: Store embeddings using FAISS (fast similarity search)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Step 4: Define a retrieval function
def retrieve(query, k=2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    results = [documents[i] for i in indices[0]]
    return results

# Test retrieval
query = "Which animal is the biggest?"
top_docs = retrieve(query)

print("Query:", query)
print("\nTop relevant documents:")
for doc in top_docs:
    print("-", doc)
