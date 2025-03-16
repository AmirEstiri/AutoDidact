import json
import pickle
from langchain_community.vectorstores import FAISS
from embeddings import CustomHuggingFaceEmbeddings
from langchain.docstore.document import Document


# Convert the dictionary chunks into Document objects
chunks_dict = json.load(open("data/chunks.json"))
chunks = [
    Document(page_content=text, metadata={"id": id}) for id, text in chunks_dict.items()
]

# Save chunks for later use
with open("data/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print(f"Saved {len(chunks)} chunks to data/chunks.pkl")

embeddings = CustomHuggingFaceEmbeddings()

# Create a FAISS vector store from the document chunks and save it locally
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("data/faiss_index")
print("Saved FAISS index to 'data/faiss_index'")
