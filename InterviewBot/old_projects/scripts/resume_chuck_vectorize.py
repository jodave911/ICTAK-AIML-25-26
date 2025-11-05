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

# --- 1. JSON SCHEMA DEFINITION ---
# Define the desired structured output as a JSON Schema dictionary.

JSON_RESUME_SCHEMA = {
    "title": "Resume",
    "description": "The structured representation of a resume.",
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The full name of the candidate."},
        "summary": {"type": "string", "description": "A brief summary of the candidate's profile."},
        "work_experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "The job title or role."},
                    "company": {"type": "string", "description": "The name of the company."},
                    "start_date": {"type": "string", "description": "The start date of the employment."},
                    "end_date": {"type": "string", "description": "The end date of the employment (or 'Present')."},
                    "responsibilities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role", "company", "responsibilities"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": "string", "description": "The degree obtained."},
                    "institution": {"type": "string", "description": "The name of the institution."},
                    "graduation_date": {"type": "string", "description": "The graduation date."},
                },
                "required": ["degree", "institution"],
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "summary", "work_experience", "education", "skills"],
}

# --- 2. AGENTIC EXTRACTION (LANGCHAIN + GEMINI) ---

def extract_resume_data(resume_text: str) -> Dict:
    """
    Uses LangChain and Gemini to extract structured data from resume text
    based on a JSON schema dictionary.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert HR assistant specializing in parsing resumes. Your task is to extract relevant information"
            "from the following resume text and format it as a valid JSON object. Adhere strictly to the provided schema."
            "If a piece of information is not found, use null or an empty list."),
            ("human", "{resume_text}"),
        ]
    )


    # Chain the prompt and model with the structured output schema
    structured_runnable = prompt | llm.with_structured_output(schema=JSON_RESUME_SCHEMA)

    print("Invoking Gemini for agentic extraction...")
    result = structured_runnable.invoke({"resume_text": resume_text})
    print("Extraction complete.")
    return result

# --- 3. SEMANTIC CHUNKING ---

def create_semantic_chunks(resume_data: Dict) -> List[Document]:
    """
    Converts the structured dictionary into a list of semantic LangChain Documents.
    """
    chunks = []

    # Summary chunk
    chunks.append(Document(
        page_content=f"Summary: {resume_data.get('summary', '')}",
        metadata={"category": "summary", "name": resume_data.get('name', '')}
    ))

    # Work Experience chunks
    for job in resume_data.get('work_experience', []):
        content = (
            f"Role: {job.get('role')} at {job.get('company')} ({job.get('start_date')} - {job.get('end_date')}). "
            f"Responsibilities: {' '.join(job.get('responsibilities', []))}"
        )
        chunks.append(Document(
            page_content=content,
            metadata={"category": "work_experience", "company": job.get('company'), "role": job.get('role')}
        ))

    # Education chunks
    for edu in resume_data.get('education', []):
        content = f"Degree: {edu.get('degree')} from {edu.get('institution')} (Graduated: {edu.get('graduation_date')})."
        chunks.append(Document(
            page_content=content,
            metadata={"category": "education", "institution": edu.get('institution')}
        ))

    # Skills chunk
    skills = resume_data.get('skills', [])
    if skills:
        chunks.append(Document(
            page_content=f"Skills: {', '.join(skills)}",
            metadata={"category": "skills"}
        ))

    print(f"Created {len(chunks)} semantic chunks.")
    return chunks

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    # --- Build Phase ---
    print("--- Starting Resume Processing Pipeline ---")

    with open("txts/resume.txt", "r", encoding='utf-8') as f:
        resume_text = f.read()

    structured_resume = extract_resume_data(resume_text)
    documents = create_semantic_chunks(structured_resume)

    print("\n--- Sample Chunk ---")
    print(documents[1])
    print("--------------------\n")

    # --- 4. VECTORIZATION & STORAGE ---
    print("Initializing embedding model and FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss/faiss_resume_index")
    print("FAISS index saved locally to 'faiss_resume_index'.")
    print("--- Pipeline Completed Successfully ---\n")
