import os
from typing import List, Dict

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. CONSOLIDATED SCHEMAS ---

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


# --- 2. HELPER FUNCTIONS ---

def _extract_data(llm: ChatGoogleGenerativeAI, text: str, schema: dict, doc_type: str) -> Dict:
    """Internal function to extract structured data from text using a given schema."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an expert assistant specializing in parsing {doc_type}s. Extract relevant information and format it as a valid JSON object adhering strictly to the provided schema. If info is not found, use null or an empty list."),
        ("human", "{text_input}"),
    ])
    structured_runnable = prompt | llm.with_structured_output(schema=schema)
    print(f"Invoking Gemini for agentic extraction of {doc_type}...")
    result = structured_runnable.invoke({"text_input": text})
    print("Extraction complete.")
    return result

def _create_semantic_chunks(data: Dict, doc_type: str) -> List[Document]:
    """Internal function to create semantic chunks based on the document type."""
    chunks = []
    if doc_type == "Resume":
        # Resume chunking logic
        chunks.append(Document(page_content=f"Summary: {data.get('summary', '')}", metadata={"category": "summary", "name": data.get('name', '')}))
        for job in data.get('work_experience', []):
            content = f"Role: {job.get('role')} at {job.get('company')} ({job.get('start_date')} - {job.get('end_date')}). Responsibilities: {' '.join(job.get('responsibilities', []))}"
            chunks.append(Document(page_content=content, metadata={"category": "work_experience", "company": job.get('company'), "role": job.get('role')}))
        for edu in data.get('education', []):
            content = f"Degree: {edu.get('degree')} from {edu.get('institution')} (Graduated: {edu.get('graduation_date')})."
            chunks.append(Document(page_content=content, metadata={"category": "education", "institution": edu.get('institution')}))
        if skills := data.get('skills', []):
            chunks.append(Document(page_content=f"Skills: {', '.join(skills)}", metadata={"category": "skills"}))

    elif doc_type == "Job Description":
        # Job Description chunking logic
        company = data.get('company', 'N/A')
        job_title = data.get('job_title', 'N/A')
        overview_content = f"Job Title: {job_title} at {company}. Location: {data.get('location', 'N/A')}. Company Summary: {data.get('company_summary', '')}"
        chunks.append(Document(page_content=overview_content.strip(), metadata={"category": "overview", "company": company, "job_title": job_title}))
        if responsibilities := data.get('responsibilities', []):
            chunks.append(Document(page_content=f"Responsibilities: {' '.join(responsibilities)}", metadata={"category": "responsibilities", "company": company, "job_title": job_title}))
        if required := data.get('required_qualifications', []):
            chunks.append(Document(page_content=f"Required Qualifications: {' '.join(required)}", metadata={"category": "required_qualifications", "company": company, "job_title": job_title}))
        if preferred := data.get('preferred_qualifications', []):
            chunks.append(Document(page_content=f"Preferred Qualifications: {' '.join(preferred)}", metadata={"category": "preferred_qualifications", "company": company, "job_title": job_title}))

    print(f"Created {len(chunks)} semantic chunks for the {doc_type}.")
    return chunks

# --- 3. MAIN PUBLIC FUNCTION ---

def process_and_vectorize(file_path: str, schema: dict, index_path: str, doc_type: str):
    """
    Reads a text file, extracts structured data, creates semantic chunks,
    and saves them to a FAISS index. This is the main function for this module.
    """
    print(f"\n--- Processing {doc_type} from {file_path} ---")

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

    # Read text from file
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()

    # Execute the processing pipeline
    structured_data = _extract_data(llm, text, schema, doc_type)
    documents = _create_semantic_chunks(structured_data, doc_type)

    if not documents:
        raise ValueError(f"No documents were created for {doc_type}. Check chunking logic.")

    # Vectorize and store
    print(f"Initializing embedding model for {doc_type}...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    db.save_local(index_path)

    print(f"FAISS index for {doc_type} saved to '{index_path}'.")