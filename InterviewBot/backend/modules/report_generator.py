import os
import json
import logging

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. HELPER FUNCTIONS ---

def _setup_logging(log_file: str):
    """Configures the logger to write LLM interactions to a specific file."""
    # Remove any existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    print(f"Logging is configured. LLM interactions will be saved to '{log_file}'.")

def _setup_environment(jd_index_path: str, resume_index_path: str, transcript_path: str):
    """Initializes models and loads data sources from specified paths."""
    print("Initializing models and loading data sources for report generation...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    jd_db = FAISS.load_local(jd_index_path, embeddings, allow_dangerous_deserialization=True)
    resume_db = FAISS.load_local(resume_index_path, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Interview transcript not found at '{transcript_path}'.")
    with open(transcript_path, "r", encoding='utf-8') as f:
        transcript = json.load(f)
    if not transcript:
        raise ValueError("Interview transcript is empty.")

    print("Setup complete. All data sources loaded.")
    return llm, jd_db, resume_db, transcript

def _log_and_invoke_llm(chain, inputs: dict, purpose: str):
    """Logs the request and response of an LLM invocation."""
    logging.info(f"--- LLM Request Start ---")
    logging.info(f"Purpose: {purpose}")
    logging.info(f"Input Context Length: {len(inputs.get('context', ''))} characters")
    response = chain.invoke(inputs)
    response_content = response.content if hasattr(response, 'content') else str(response)
    logging.info(f"Output: {response_content}")
    logging.info(f"--- LLM Request End ---\n")
    return response

def _gather_comprehensive_context(jd_db: FAISS, resume_db: FAISS, transcript: list) -> str:
    """Formats all data sources into a single context string for the LLM."""
    print("Aggregating context from all sources...")
    transcript_str = "--- INTERVIEW TRANSCRIPT ---\n"
    for i, item in enumerate(transcript):
        transcript_str += f"Question {i+1} (Type: {item['type']}): {item['question']}\n"
        transcript_str += f"Candidate's Answer: {item['answer']}\n\n"

    jd_docs = jd_db.similarity_search("key responsibilities and all required qualifications", k=5)
    jd_context_str = "--- KEY JOB DESCRIPTION INFO ---\n"
    jd_context_str += "\n".join([doc.page_content for doc in jd_docs])

    resume_docs = resume_db.similarity_search("candidate's summary, work experience, and skills", k=5)
    resume_context_str = "\n\n--- KEY RESUME INFO ---\n"
    resume_context_str += "\n".join([doc.page_content for doc in resume_docs])

    full_context = f"{jd_context_str}\n{resume_context_str}\n\n{transcript_str}"
    print("Context aggregation complete.")
    return full_context

def _generate_evaluation_report(llm: ChatGoogleGenerativeAI, context: str) -> str:
    """Generates the final candidate evaluation report using the aggregated context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert HR analyst and senior hiring manager. Your task is to write a comprehensive, objective, and well-structured candidate evaluation report. "
         "Base your entire analysis STRICTLY on the provided context. Do not invent information. Cite evidence from the transcript where appropriate."),
        ("human", "Please generate the candidate evaluation report based on the following context:\n\n{context}\n\nUse the following Markdown format for the report:\n\n# Candidate Evaluation Report\n\n"
         "## 1. Overall Summary\n*Provide a concise, high-level summary of the candidate's background, interview performance, and overall fit for the role.*\n\n"
         "## 2. Alignment with Job Requirements\n*Analyze how the candidate's skills and experience align with the key requirements from the job description.*\n\n"
         "### Strengths\n*   **Requirement 1:** [Describe how the candidate meets this requirement, citing specific examples.]\n*   **Requirement 2:** [Describe how the candidate meets this requirement...]\n\n"
         "### Weaknesses / Gaps\n*   **Requirement 1:** [Describe where the candidate shows a gap or weakness.]\n*   **Requirement 2:** [Describe another potential gap...]\n\n"
         "## 3. Technical and Project-Specific Skills Evaluation\n*Evaluate the depth and practicality of the candidate's technical skills as demonstrated in the interview. Assess their problem-solving abilities and the clarity of their technical explanations.*\n\n"
         "## 4. Communication Skills\n*Assess the candidate's ability to articulate their thoughts clearly and concisely. Did they understand the questions? Were their answers structured and easy to follow?*\n\n"
         "## 5. Potential Red Flags\n*Identify any potential concerns. This could include inconsistencies between the resume and interview, evasiveness in answers, or significant gaps in core knowledge. If no red flags are identified, state 'None identified'.*\n\n"
         "## 6. Analysing the Questions and Answers\n*Identify any potential questions. Strictly analyse the user answers to the questions which was prompted. Evaluate and give a breif evaluation of each question with answer and the Justiifiction to the question'.*\n\n"
         "## 7. Final Recommendation and Justification\n"
         "**Recommendation:** [Choose one: Strongly Recommend / Recommend / Recommend with Reservations / Do Not Recommend]\n\n"
         "**Justification:**\n*Provide a clear, evidence-based justification for your recommendation, summarizing the most critical factors from the sections above that led to your decision.*")
    ])
    chain = prompt | llm
    print("Generating the final report... This may take a moment.")
    response = _log_and_invoke_llm(chain, {"context": context}, "Generate Full Candidate Report")
    return response.content

# --- 2. MAIN PUBLIC FUNCTION ---

def create_report(jd_index_path: str, resume_index_path: str, transcript_path: str, output_path: str, log_file: str):
    """
    Main function for this module. Generates the final candidate report by aggregating
    all data sources and using an LLM for analysis.
    """
    try:
        _setup_logging(log_file)
        llm, jd_db, resume_db, transcript = _setup_environment(jd_index_path, resume_index_path, transcript_path)

        comprehensive_context = _gather_comprehensive_context(jd_db, resume_db, transcript)
        report = _generate_evaluation_report(llm, comprehensive_context)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print("\n" + "="*50)
        print("      CANDIDATE EVALUATION REPORT GENERATED")
        print("="*50 + "\n")
        print(report)
        print(f"\nReport has been saved to '{output_path}'")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError during report generation: {e}")
        print("Report generation aborted.")