import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

load_dotenv()

# === 1. Load documents ===
resume_loader = TextLoader("resume_candidate_a.txt", encoding="utf-8")
job_loader = TextLoader("jd_genai_engineer.txt", encoding="utf-8")

resume_docs = resume_loader.load()
job_docs = job_loader.load()

# Add metadata to distinguish sources
for doc in resume_docs:
    doc.metadata["source"] = "resume"
for doc in job_docs:
    doc.metadata["source"] = "job_description"

all_docs = resume_docs + job_docs

# === 2. Semantic Chunking (Agentic-style) ===
# Use smaller chunk_size to preserve job/resume sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True
)

chunks = text_splitter.split_documents(all_docs)

# Optional: Add chunk index for debugging
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i

print(f"✅ Created {len(chunks)} chunks")

# === 3. Embed & Store in FAISS ===
embeddings = HuggingFaceEmbeddings(model="hkunlp/instructor-large")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save for later (optional)
# vectorstore.save_local("faiss_resume_job_index")

# === 4. Test Retrieval ===
print(chunks.)
# query = "What Python and AWS experience does the candidate have, and does it match the job requirements?"

# # Get top 5 most relevant chunks with scores
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=5)

# print("\n🔍 Retrieval Results:\n")
# for i, (doc, score) in enumerate(docs_and_scores):
#     print(f"--- Chunk {i+1} (Score: {score:.4f}) ---")
#     print(f"Source: {doc.metadata['source']}")
#     print(f"Content: {doc.page_content[:300]}...\n")