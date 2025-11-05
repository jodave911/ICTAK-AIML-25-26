import os
import json
import logging
from dotenv import load_dotenv

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SETUP ---
FAISS_JD_PATH = "faiss/faiss_jd_index"
FAISS_RESUME_PATH = "faiss/faiss_resume_index"
TRANSCRIPT_PATH = "interview_transcript_fullscore.json"

def setup_logging():
    """Configures the logger to write LLM interactions to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='report_generation.log',
        filemode='a'  # Append to the log file on each run
    )
    print("Logging is configured. LLM interactions will be saved to 'report_generation.log'.")

def setup_environment():
    """Loads environment variables, models, vector stores, and the interview transcript."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    print("Initializing models and loading data sources...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    jd_db = FAISS.load_local(FAISS_JD_PATH, embeddings, allow_dangerous_deserialization=True)
    resume_db = FAISS.load_local(FAISS_RESUME_PATH, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(TRANSCRIPT_PATH):
        raise FileNotFoundError(f"Interview transcript not found at '{TRANSCRIPT_PATH}'.")
    with open(TRANSCRIPT_PATH, "r", encoding='utf-8') as f:
        transcript = json.load(f)
    if not transcript:
        raise ValueError("Interview transcript is empty.")

    print("Setup complete. All data sources loaded.")
    return llm, jd_db, resume_db, transcript

# --- 2. LOGGING WRAPPER ---
def log_and_invoke_llm(chain, inputs: dict, purpose: str):
    """Logs the request and response of an LLM invocation."""
    logging.info(f"--- LLM Request Start ---")
    logging.info(f"Purpose: {purpose}")
    logging.info(f"Input Context Length: {len(inputs.get('context', ''))} characters")

    response = chain.invoke(inputs)

    response_content = response.content if hasattr(response, 'content') else str(response)
    logging.info(f"Output: {response_content}")
    logging.info(f"--- LLM Request End ---\n")

    return response

# --- 3. RAG CONTEXT AGGREGATION (Unchanged) ---
def gather_comprehensive_context(jd_db: FAISS, resume_db: FAISS, transcript: list) -> str:
    print("Aggregating context from all sources...")
    transcript_str = "--- INTERVIEW TRANSCRIPT ---\n"
    for i, item in enumerate(transcript):
        transcript_str += f"Question {i+1} (Type: {item['type']}): {item['question']}\n"
        transcript_str += f"Candidate's Answer: {item['answer']}\n\n"

    jd_docs = jd_db.similarity_search("key responsibilities and all required qualifications", k=4)
    jd_context_str = "--- KEY JOB DESCRIPTION INFO ---\n"
    jd_context_str += "\n".join([doc.page_content for doc in jd_docs])

    resume_docs = resume_db.similarity_search("candidate's summary, work experience, and skills", k=4)
    resume_context_str = "\n\n--- KEY RESUME INFO ---\n"
    resume_context_str += "\n".join([doc.page_content for doc in resume_docs])

    full_context = f"{jd_context_str}\n{resume_context_str}\n\n{transcript_str}"
    print("Context aggregation complete.")
    return full_context

# --- 4. REPORT GENERATION (Prompt Enhanced) ---
def generate_evaluation_report(llm: ChatGoogleGenerativeAI, context: str) -> str:
    """Generates the final candidate evaluation report using the aggregated context."""

    # --- MODIFIED PROMPT ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert HR analyst and senior hiring manager. Your task is to write a comprehensive, objective, and well-structured candidate evaluation report. "
         "Base your entire analysis STRICTLY on the provided context, which includes the job description, the candidate's resume, and the full interview transcript. "
         "Do not invent information. Cite evidence from the transcript where appropriate."),
        ("human", "Please generate the candidate evaluation report based on the following context:\n\n"
         "{context}\n\n"
         "Use the following Markdown format for the report:\n\n"
         "# Candidate Evaluation Report\n\n"
         "## 1. Overall Summary\n"
         "*Provide a concise, high-level summary of the candidate's background, interview performance, and overall fit for the role.*\n\n"
         "## 2. Alignment with Job Requirements\n"
         "*Analyze how the candidate's skills and experience align with the key requirements from the job description.*\n\n"
         "### Strengths\n"
         "*   **Requirement 1:** [Describe how the candidate meets this requirement, citing specific examples.]\n"
         "*   **Requirement 2:** [Describe how the candidate meets this requirement...]\n\n"
         "### Weaknesses / Gaps\n"
         "*   **Requirement 1:** [Describe where the candidate shows a gap or weakness.]\n"
         "*   **Requirement 2:** [Describe another potential gap...]\n\n"
         "## 3. Technical and Project-Specific Skills Evaluation\n"
         "*Evaluate the depth and practicality of the candidate's technical skills as demonstrated in the interview. Assess their problem-solving abilities and the clarity of their technical explanations.*\n\n"
         "## 4. Communication Skills\n"
         "*Assess the candidate's ability to articulate their thoughts clearly and concisely. Did they understand the questions? Were their answers structured and easy to follow?*\n\n"
         "## 5. Potential Red Flags\n"
         "*Identify any potential concerns. This could include inconsistencies between the resume and interview, evasiveness in answers, or significant gaps in core knowledge. If no red flags are identified, state 'None identified'.*\n\n"
         "## 6. Final Recommendation and Justification\n"
         "**Recommendation:** [Choose one: Strongly Recommend / Recommend / Recommend with Reservations / Do Not Recommend]\n\n"
         "**Justification:**\n"
         "*Provide a clear, evidence-based justification for your recommendation, summarizing the most critical factors from the sections above that led to your decision.*")
    ])

    chain = prompt | llm
    print("Generating the final report... This may take a moment.")
    # --- LOGGING ADDED ---
    response = log_and_invoke_llm(chain, {"context": context}, "Generate Full Candidate Report")
    return response.content

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        setup_logging() # Initialize the logger
        llm, jd_db, resume_db, transcript = setup_environment()

        comprehensive_context = gather_comprehensive_context(jd_db, resume_db, transcript)
        report = generate_evaluation_report(llm, comprehensive_context)

        report_filename = "candidate_evaluation_report.md"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report)

        print("\n" + "="*50)
        print("      CANDIDATE EVALUATION REPORT GENERATED")
        print("="*50 + "\n")
        print(report)
        print("\n" + "="*50)
        print(f"Report has been saved to '{report_filename}'")
        print("="*50)

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        print("Report generation aborted.")