import streamlit as st
import os
import json
from dotenv import load_dotenv

from backend.modules.data_processor import process_and_vectorize, JSON_JD_SCHEMA, JSON_RESUME_SCHEMA
from backend.modules.interview_bot import InterviewBot 
from backend.modules.report_generator import create_report

# --- 1. CENTRALIZED CONFIGURATION ---
BASE_DIR = "backend"
TXT_DIR = os.path.join(BASE_DIR, "txts")
FAISS_DIR = os.path.join(BASE_DIR, "faiss")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

JD_TXT_PATH = os.path.join(TXT_DIR, "job_description.txt")
RESUME_TXT_PATH = os.path.join(TXT_DIR, "resume.txt")
FAISS_JD_PATH = os.path.join(FAISS_DIR, "faiss_jd_index")
FAISS_RESUME_PATH = os.path.join(FAISS_DIR, "faiss_resume_index")
TRANSCRIPT_PATH = os.path.join(TRANSCRIPT_DIR, "interview_transcript.json")
REPORT_PATH = os.path.join(BASE_DIR, "candidate_evaluation_report.md")
INTERVIEW_LOG_PATH = os.path.join(LOG_DIR, "llm_usage.log")
REPORT_LOG_PATH = os.path.join(LOG_DIR, "report_generation.log")

# --- 2. STREAMLIT APP SETUP ---
st.set_page_config(page_title="AI Interview & Analysis Pipeline", layout="wide")
st.title("🤖 AI Interview & Analysis Pipeline")

if 'pipeline_step' not in st.session_state:
    st.session_state.pipeline_step = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interview_bot" not in st.session_state:
    st.session_state.interview_bot = None

with st.sidebar:
    st.header("Configuration")
    load_dotenv()
    api_key_input = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("Please enter your Google API Key to proceed.")
        st.stop()
    else:
        st.success("API Key configured.")

# --- 3. PIPELINE STEPS IN UI ---

st.header("Step 1: Process Job Description and Resume")
if st.session_state.pipeline_step == 0:
    jd_file = st.file_uploader("Upload Job Description (.txt)", type="txt")
    resume_file = st.file_uploader("Upload Candidate Resume (.txt)", type="txt")

    if st.button("Process Documents", disabled=(not jd_file or not resume_file)):
        try:
            with open(JD_TXT_PATH, "w", encoding="utf-8") as f: f.write(jd_file.getvalue().decode("utf-8"))
            with open(RESUME_TXT_PATH, "w", encoding="utf-8") as f: f.write(resume_file.getvalue().decode("utf-8"))
            with st.spinner("Processing and vectorizing documents..."):
                process_and_vectorize(JD_TXT_PATH, JSON_JD_SCHEMA, FAISS_JD_PATH, "Job Description")
                process_and_vectorize(RESUME_TXT_PATH, JSON_RESUME_SCHEMA, FAISS_RESUME_PATH, "Resume")
            st.session_state.pipeline_step = 1
            st.success("Documents processed successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during document processing: {e}")

if st.session_state.pipeline_step > 0:
    st.success("✅ Step 1: Documents Processed Successfully")

st.header("Step 2: Conduct AI Interview")
if st.session_state.pipeline_step == 1:
    num_questions = st.number_input("Enter the number of questions for the interview", min_value=1, max_value=10, value=5, step=1)

    if st.button("Start Interview"):
        try:
            st.session_state.interview_bot = InterviewBot(
                jd_index_path=FAISS_JD_PATH,
                resume_index_path=FAISS_RESUME_PATH,
                log_file=INTERVIEW_LOG_PATH,
                num_questions=num_questions
            )
            initial_question = st.session_state.interview_bot.start_interview()
            st.session_state.messages = [{"role": "assistant", "content": initial_question}]
            st.rerun()
        except Exception as e:
            st.error(f"Failed to initialize the interview bot: {e}")

    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Your answer..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.interview_bot.process_user_answer(prompt)
                    if response == "END_OF_INTERVIEW":
                        st.session_state.pipeline_step = 2
                        st.session_state.interview_bot.save_transcript(TRANSCRIPT_PATH)
                        final_message = "That's all the questions I have for you. Thank you for your time. The interview is now complete."
                        st.session_state.messages.append({"role": "assistant", "content": final_message})
                        st.success("Interview finished and transcript saved.")
                        st.rerun()
                    else:
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

if st.session_state.pipeline_step > 1:
    st.success("✅ Step 2: Interview Completed")

st.header("Step 3: Generate Candidate Evaluation Report")
if st.session_state.pipeline_step == 2:
    if st.button("Generate Report"):
        try:
            with st.spinner("Generating the final evaluation report..."):
                create_report(FAISS_JD_PATH, FAISS_RESUME_PATH, TRANSCRIPT_PATH, REPORT_PATH, REPORT_LOG_PATH)
            st.session_state.pipeline_step = 3
            st.success("Report generated successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during report generation: {e}")

if st.session_state.pipeline_step > 2:
    st.success("✅ Step 3: Report Generated Successfully")
    st.header("Final Candidate Evaluation Report")
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report_content = f.read()
        st.markdown(report_content)
        st.download_button("Download Report as MD", report_content, "candidate_evaluation_report.md", "text/markdown")
    except FileNotFoundError:
        st.error("The report file was not found. Please try generating it again.")