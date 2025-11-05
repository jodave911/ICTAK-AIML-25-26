import os
import json
import uuid
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your refactored module functions
from modules.data_processor import process_and_vectorize, JSON_JD_SCHEMA, JSON_RESUME_SCHEMA
from modules.interview_bot import run_interview_session # We'll borrow helpers from this
from modules.report_generator import create_report # And this

# --- Application Setup ---
app = FastAPI(
    title="AI Interview Backend",
    description="An API for conducting AI-powered candidate interviews.",
    version="1.0.0"
)

# This allows your frontend (running on localhost:5173) to communicate with this backend.
# IMPORTANT: In production, you should restrict this to your actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for now, change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management ---
# In-memory dictionary to store session data.
# In a production environment, this should be replaced with a database like Redis.
SESSIONS = {}

# --- Pydantic Models for Request Bodies ---
# These models ensure that the data sent to your API has the correct structure and types.
class StartInterviewRequest(BaseModel):
    jd_text: str
    resume_text: str
    num_questions: int

class ChatRequest(BaseModel):
    answer: str

# --- API Endpoints ---

@app.post("/interviews", status_code=201)
async def start_interview(request: StartInterviewRequest):
    """
    Starts a new interview session.
    1. Generates a unique ID.
    2. Creates temporary files for the JD and Resume.
    3. Processes and vectorizes them into session-specific FAISS indexes.
    4. Initializes the session state.
    """
    interview_id = str(uuid.uuid4())
    session_path = os.path.join("data", "sessions", interview_id)
    os.makedirs(session_path, exist_ok=True)

    # Write the text content to temporary files for processing
    jd_txt_path = os.path.join(session_path, "jd.txt")
    resume_txt_path = os.path.join(session_path, "resume.txt")
    with open(jd_txt_path, "w", encoding="utf-8") as f: f.write(request.jd_text)
    with open(resume_txt_path, "w", encoding="utf-8") as f: f.write(request.resume_text)

    jd_index_path = os.path.join(session_path, "faiss_jd_index")
    resume_index_path = os.path.join(session_path, "faiss_resume_index")

    try:
        process_and_vectorize(jd_txt_path, JSON_JD_SCHEMA, jd_index_path, "Job Description")
        process_and_vectorize(resume_txt_path, JSON_RESUME_SCHEMA, resume_index_path, "Resume")
    except Exception as e:
        # Clean up if processing fails
        shutil.rmtree(session_path)
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")

    # Initialize session state
    SESSIONS[interview_id] = {
        "jd_index_path": jd_index_path,
        "resume_index_path": resume_index_path,
        "transcript_path": os.path.join(session_path, "transcript.json"),
        "num_questions": request.num_questions,
        "questions_asked": 0,
        "transcript": []
    }

    # For now, we'll just return the ID. The frontend will then call /chat to get the first question.
    return {
        "interview_id": interview_id,
        "message": "Session created successfully. Ready to start interview."
    }


@app.post("/interviews/{interview_id}/chat")
async def chat_turn(interview_id: str, request: ChatRequest):
    """
    Handles a single turn of the interview.
    - On the first call (answer="start"), it generates the first question.
    - On subsequent calls, it records the answer, evaluates it, and generates the next question.
    """
    if interview_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    session = SESSIONS[interview_id]

    # This is a placeholder for the full, complex logic from your interview_bot module.
    # For a real implementation, you would move the question generation and evaluation
    # logic into this endpoint.

    # Record the previous answer if a question has been asked
    if session["questions_asked"] > 0 and request.answer:
        session["transcript"][-1]["answer"] = request.answer

    # Check if the interview is over
    if session["questions_asked"] >= session["num_questions"]:
        # Save the final transcript
        with open(session["transcript_path"], "w") as f:
            json.dump(session["transcript"], f, indent=4)
        return {
            "question": "That's all the questions I have. Thank you for attending the Interview.",
            "interview_over": True
        }

    # Generate the next question (simplified for API clarity)
    # In a real app, you would call your `_generate_..._question` functions here.
    session["questions_asked"] += 1
    next_question = f"This is question {session['questions_asked']}/{session['num_questions']}. Please describe your experience with FastAPI."

    session["transcript"].append({"type": "SITUATIONAL", "question": next_question, "answer": ""})

    return {
        "question": next_question,
        "interview_over": False
    }


@app.post("/interviews/{interview_id}/report")
async def get_report(interview_id: str):
    """
    Generates and returns the final evaluation report for a completed interview.
    """
    if interview_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    session = SESSIONS[interview_id]

    # In a real app, you would call your full report_generator logic here.
    # For now, we'll generate a simplified report.
    try:
        # We'll use a simplified version of your report generator for this example.
        # A full implementation would call `create_report` and read the file.
        report_content = f"# Candidate Evaluation Report for Interview {interview_id}\n\n"
        report_content += "## Interview Transcript\n\n"
        for item in session['transcript']:
            report_content += f"**Question:** {item['question']}\n\n"
            report_content += f"**Answer:** {item['answer']}\n\n---\n\n"

        # Clean up the session data after generating the report
        shutil.rmtree(os.path.dirname(session["jd_index_path"]))
        del SESSIONS[interview_id]

        return {"report": report_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
