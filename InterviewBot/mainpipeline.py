import os
from dotenv import load_dotenv

# --- Import the main functions and schemas from our new modules ---
from backend.modules.data_processor import process_and_vectorize, JSON_JD_SCHEMA, JSON_RESUME_SCHEMA
from backend.modules.interview_bot import run_interview_session
from backend.modules.report_generator import create_report

# --- 1. CENTRALIZED CONFIGURATION ---
# Define all paths in one place for easy management.

# Input text files
JD_TXT_PATH = "backend/txts/job_description.txt"
RESUME_TXT_PATH = "backend/txts/resume.txt"

# Output FAISS vector stores
FAISS_JD_PATH = "backend/faiss/faiss_jd_index"
FAISS_RESUME_PATH = "backend/faiss/faiss_resume_index"

# Intermediate and final output files
TRANSCRIPT_PATH = "backend/transcripts/interview_transcript.json"
REPORT_PATH = "backend/candidate_evaluation_report.md"

# Log files
INTERVIEW_LOG_PATH = "backend/logs/llm_usage.log"
REPORT_LOG_PATH = "backend/logs/report_generation.log"


def main():
    """
    Executes the entire AI Interview and Analysis pipeline from start to finish.
    """
    # Load environment variables from .env file
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

    print("==============================================")
    print("  STARTING AI INTERVIEW & ANALYSIS PIPELINE   ")
    print("==============================================")

    try:
        # --- STEP 1: Processing and Vectorizing Documents ---
        print("\n--- STEP 1: Processing and Vectorizing Documents ---")

        # Ensure the FAISS directory exists to prevent errors
        os.makedirs("faiss", exist_ok=True)

        process_and_vectorize(
            file_path=JD_TXT_PATH,
            schema=JSON_JD_SCHEMA,
            index_path=FAISS_JD_PATH,
            doc_type="Job Description"
        )

        process_and_vectorize(
            file_path=RESUME_TXT_PATH,
            schema=JSON_RESUME_SCHEMA,
            index_path=FAISS_RESUME_PATH,
            doc_type="Resume"
        )
        print("\n--- STEP 1 COMPLETED: Documents processed successfully. ---")

        # --- STEP 2: Conducting the Interactive Interview ---
        print("\n--- STEP 2: Conducting the Interactive Interview ---")
        run_interview_session(
            jd_index_path=FAISS_JD_PATH,
            resume_index_path=FAISS_RESUME_PATH,
            transcript_path=TRANSCRIPT_PATH,
            log_file=INTERVIEW_LOG_PATH
        )
        print("\n--- STEP 2 COMPLETED: Interview finished. ---")

        # --- STEP 3: Generating the Final Evaluation Report ---
        print("\n--- STEP 3: Generating the Final Evaluation Report ---")
        create_report(
            jd_index_path=FAISS_JD_PATH,
            resume_index_path=FAISS_RESUME_PATH,
            transcript_path=TRANSCRIPT_PATH,
            output_path=REPORT_PATH,
            log_file=REPORT_LOG_PATH
        )
        print("\n--- STEP 3 COMPLETED: Report generated. ---")

        print("\n==============================================")
        print("    PIPELINE COMPLETED SUCCESSFULLY!    ")
        print(f"Final report is available at: {REPORT_PATH}")
        print("==============================================")

    except Exception as e:
        print(f"\n\n!!! AN ERROR OCCURRED DURING PIPELINE EXECUTION !!!")
        print(f"Error: {e}")
        print("Pipeline aborted.")

if __name__ == "__main__":
    main()