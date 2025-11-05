import os
import json
from typing import List, Dict
from dotenv import load_dotenv

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. JSON SCHEMA DEFINITION FOR JOB DESCRIPTION ---
# Define the desired structured output for a job description.

JSON_JD_SCHEMA = {
    "title": "JobDescription",
    "description": "The structured representation of a job description.",
    "type": "object",
    "properties": {
        "job_title": {"type": "string", "description": "The title of the job position."},
        "company": {"type": "string", "description": "The name of the company hiring."},
        "location": {"type": "string", "description": "The location of the job (e.g., city, state, remote)."},
        "company_summary": {"type": "string", "description": "A brief summary of the company."},
        "responsibilities": {
            "type": "array",
            "description": "A list of key responsibilities for the role.",
            "items": {"type": "string"},
        },
        "required_qualifications": {
            "type": "array",
            "description": "A list of essential qualifications and skills.",
            "items": {"type": "string"},
        },
        "preferred_qualifications": {
            "type": "array",
            "description": "A list of desired but not essential qualifications.",
            "items": {"type": "string"},
        },
    },
    "required": ["job_title", "company", "responsibilities", "required_qualifications"],
}

# --- 2. AGENTIC EXTRACTION (LANGCHAIN + GEMINI) ---

def extract_jd_data(jd_text: str) -> Dict:
    """
    Uses LangChain and Gemini to extract structured data from job description text
    based on a JSON schema dictionary.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert recruiting assistant specializing in parsing job descriptions. "
             "Your task is to extract relevant information from the following job description text and format it as a valid JSON object. "
             "Adhere strictly to the provided schema. If a piece of information is not found, use null or an empty list."),
            ("human", "{jd_text}"),
        ]
    )

    # Chain the prompt and model with the structured output schema
    structured_runnable = prompt | llm.with_structured_output(schema=JSON_JD_SCHEMA)

    print("Invoking Gemini for agentic extraction of Job Description...")
    result = structured_runnable.invoke({"jd_text": jd_text})
    print("Extraction complete.")
    return result

# --- 3. SEMANTIC CHUNKING ---

def create_jd_semantic_chunks(jd_data: Dict) -> List[Document]:
    """
    Converts the structured JD dictionary into a list of semantic LangChain Documents.
    """
    chunks = []
    company = jd_data.get('company', 'N/A')
    job_title = jd_data.get('job_title', 'N/A')

    # Job Overview chunk
    overview_content = (
        f"Job Title: {job_title} at {company}. "
        f"Location: {jd_data.get('location', 'N/A')}. "
        f"Company Summary: {jd_data.get('company_summary', '')}"
    )
    chunks.append(Document(
        page_content=overview_content.strip(),
        metadata={"category": "overview", "company": company, "job_title": job_title}
    ))

    # Responsibilities chunk
    responsibilities = jd_data.get('responsibilities', [])
    if responsibilities:
        chunks.append(Document(
            page_content=f"Responsibilities: {' '.join(responsibilities)}",
            metadata={"category": "responsibilities", "company": company, "job_title": job_title}
        ))

    # Required Qualifications chunk
    required_qualifications = jd_data.get('required_qualifications', [])
    if required_qualifications:
        chunks.append(Document(
            page_content=f"Required Qualifications: {' '.join(required_qualifications)}",
            metadata={"category": "required_qualifications", "company": company, "job_title": job_title}
        ))

    # Preferred Qualifications chunk
    preferred_qualifications = jd_data.get('preferred_qualifications', [])
    if preferred_qualifications:
        chunks.append(Document(
            page_content=f"Preferred Qualifications: {' '.join(preferred_qualifications)}",
            metadata={"category": "preferred_qualifications", "company": company, "job_title": job_title}
        ))

    print(f"Created {len(chunks)} semantic chunks for the job description.")
    return chunks

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # --- Build Phase ---
    print("--- Starting Job Description Processing Pipeline ---")

    with open("txts/job_description.txt", "r", encoding='utf-8') as f:
        jd_text = f.read()

    structured_jd = extract_jd_data(jd_text)
    documents = create_jd_semantic_chunks(structured_jd)

    print("\n--- Sample Chunk ---")
    if documents:
        print(documents[1]) # Print the responsibilities chunk
    print("--------------------\n")

    # --- 4. VECTORIZATION & STORAGE ---
    print("Initializing embedding model and FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss/faiss_jd_index")
    print("FAISS index saved locally to 'faiss_jd_index'.")
    print("--- Pipeline Completed Successfully ---\n")