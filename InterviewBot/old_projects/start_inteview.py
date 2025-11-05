import os
import json
import random
import logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SETUP ---
FAISS_JD_PATH = "faiss/faiss_jd_index"
FAISS_RESUME_PATH = "faiss/faiss_resume_index"

def setup_logging():
    """Configures the logger to write LLM interactions to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='llm_usage.log',
        filemode='a'  # Append to the log file on each run
    )
    print("Logging is configured. LLM interactions will be saved to 'llm_usage.log'.")

def setup_environment():
    """Loads environment variables, initializes models and vector stores."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    print("Initializing models and loading vector stores...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(FAISS_JD_PATH) or not os.path.exists(FAISS_RESUME_PATH):
        raise FileNotFoundError("FAISS index directories not found. Please run the processing scripts first.")

    jd_db = FAISS.load_local(FAISS_JD_PATH, embeddings, allow_dangerous_deserialization=True)
    resume_db = FAISS.load_local(FAISS_RESUME_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Setup complete.")
    return llm, jd_db, resume_db

# --- 2. LOGGING WRAPPER FOR LLM INVOCATION ---

def log_and_invoke_llm(chain, inputs: dict, purpose: str):
    """Logs the request and response of an LLM invocation."""
    logging.info(f"--- LLM Request Start ---")
    logging.info(f"Purpose: {purpose}")
    # Use json.dumps for clean, multi-line logging of the input dictionary
    logging.info(f"Input: {json.dumps(inputs, indent=2)}")

    response = chain.invoke(inputs)

    # The response object is often an AIMessage, so we access its .content attribute
    response_content = response.content if hasattr(response, 'content') else str(response)
    logging.info(f"Output: {response_content}")
    logging.info(f"--- LLM Request End ---\n")

    return response

# --- 3. DIVERSE QUESTION GENERATION LOGIC (Refactored to use the logger) ---


def generate_rag_question(llm: ChatGoogleGenerativeAI, jd_topic: Document, resume_db: FAISS) -> str:
    resume_context_docs = resume_db.similarity_search(jd_topic.page_content, k=1)
    resume_context = resume_context_docs[0].page_content if resume_context_docs else "No specific context found."
    # --- MODIFIED PROMPT ---
    prompt = ChatPromptTemplate.from_template(
        "You are an expert technical interviewer. Create a SINGLE, FOCUSED behavioral question that connects a job requirement to the candidate's resume. "
        "DO NOT ask multiple questions or use numbered lists.\n\n"
        "JOB REQUIREMENT: {jd_context}\n"
        "CANDIDATE'S RESUME SNIPPET: {resume_context}\n\n"
        "Single Question (e.g., 'The role requires experience in X. I see on your resume you did Y. Can you describe a specific challenge you faced in that project?'):"
    )
    chain = prompt | llm
    inputs = {"jd_context": jd_topic.page_content, "resume_context": resume_context}
    response = log_and_invoke_llm(chain, inputs, "Generate RAG Question")
    return response.content

def generate_jd_question(llm: ChatGoogleGenerativeAI, jd_topic: Document) -> str:
    # --- MODIFIED PROMPT ---
    prompt = ChatPromptTemplate.from_template(
        "You are an expert interviewer. Based on the following job requirement, ask ONE concise question to gauge the candidate's understanding. "
        "The question must be a single sentence. Do not use lists.\n\n"
        "JOB REQUIREMENT: {jd_context}\n\n"
        "Single Question:"
    )
    chain = prompt | llm
    inputs = {"jd_context": jd_topic.page_content}
    response = log_and_invoke_llm(chain, inputs, "Generate JD-focused Question")
    return response.content

def generate_resume_question(llm: ChatGoogleGenerativeAI, resume_topic: Document) -> str:
    # --- MODIFIED PROMPT ---
    prompt = ChatPromptTemplate.from_template(
        "You are an expert interviewer. Based on this snippet from the candidate's resume, ask ONE detailed question to probe their experience. "
        "The question must be a single, focused query.\n\n"
        "RESUME SNIPPET: {resume_context}\n\n"
        "Single Question (e.g., 'Tell me more about your specific role in the X project mentioned here.'):"
    )
    chain = prompt | llm
    inputs = {"resume_context": resume_topic.page_content}
    response = log_and_invoke_llm(chain, inputs, "Generate Resume-focused Question")
    return response.content

def generate_situational_question(llm: ChatGoogleGenerativeAI, jd_topic: Document) -> str:
    # --- MODIFIED PROMPT ---
    prompt = ChatPromptTemplate.from_template(
        "You are a hiring manager. Based on the following job responsibility, create ONE practical, hypothetical scenario question to test problem-solving skills. "
        "The output must be a single, concise question. DO NOT use bullet points or numbered lists. Focus on one specific aspect of the scenario.\n\n"
        "JOB RESPONSIBILITY: {jd_context}\n\n"
        "Single Question (start with 'Imagine...' or 'Suppose...'):"
    )
    chain = prompt | llm
    inputs = {"jd_context": jd_topic.page_content}
    response = log_and_invoke_llm(chain, inputs, "Generate Situational Question")
    return response.content

# --- 4. ANSWER EVALUATION LOGIC (Refactored to use the logger) ---

def evaluate_answer(llm: ChatGoogleGenerativeAI, question: str, answer: str) -> Tuple[str, str]:
    """Evaluates the candidate's answer and decides if a follow-up is needed."""
    prompt = ChatPromptTemplate.from_template(
        """You are a senior interviewer evaluating a candidate's answer.
        Your goal is to assess if the answer has sufficient detail, examples, and substance.

        Original Question: {question}
        Candidate's Answer: {answer}

        Analyze the answer. A good answer provides specific examples, details the outcome, and clearly addresses the question. A weak answer is vague, too short, or avoids the question.

        First, on a new line, write your decision: 'SUFFICIENT' or 'INSUFFICIENT'.
        If the decision is 'INSUFFICIENT', on the next line, write a concise follow-up question to probe for more detail.
        If the answer is 'SUFFICIENT', write 'None' on the second line.

        Format:
        Decision: [SUFFICIENT/INSUFFICIENT]
        Follow-up: [Your follow-up question or None]
        """
    )
    chain = prompt | llm
    inputs = {"question": question, "answer": answer}
    response = log_and_invoke_llm(chain, inputs, "Evaluate Candidate Answer")

    try:
        lines = response.content.strip().split('\n')
        decision = lines[0].replace("Decision:", "").strip()
        follow_up = lines[1].replace("Follow-up:", "").strip()
        if follow_up.lower() == 'none':
            follow_up = None
        return decision, follow_up
    except IndexError:
        return "SUFFICIENT", None

# --- 5. MAIN INTERVIEW SESSION ---

def run_interview_session():
    """Orchestrates the entire interview process."""
    setup_logging() # Initialize the logger
    llm, jd_db, resume_db = setup_environment()

    num_questions = 4
    

    jd_topics = jd_db.similarity_search("key responsibilities and qualifications", k=10)
    resume_topics = resume_db.similarity_search("work experience and projects", k=10)

    question_generators = {
        "RAG": (generate_rag_question, {"jd_topic": lambda: random.choice(jd_topics), "resume_db": resume_db}),
        "SITUATIONAL": (generate_situational_question, {"jd_topic": lambda: random.choice(jd_topics)}),
        "JD": (generate_jd_question, {"jd_topic": lambda: random.choice(jd_topics)}),
        "RESUME": (generate_resume_question, {"resume_topic": lambda: random.choice(resume_topics)})
    }
    question_types = list(question_generators.keys())
    interview_transcript = []

    print("\n--- Starting Interview ---")
    print("Bot: Hello! Thank you for joining me today. I'd like to ask you a few questions.")

    for i in range(num_questions):
        q_type = random.choice(question_types)
        generator_func, kwargs_template = question_generators[q_type]

        kwargs = {key: value() if callable(value) else value for key, value in kwargs_template.items()}
        kwargs["llm"] = llm

        question = generator_func(**kwargs)

        print(f"\nBot (Question {i+1}/{num_questions}, Type: {q_type}): {question}")
        answer = input("You: ")

        decision, follow_up_question = evaluate_answer(llm, question, answer)
        if decision == "INSUFFICIENT" and follow_up_question:
            print(f"Bot: {follow_up_question}")
            follow_up_answer = input("You: ")
            question += f"\n[Follow-up Question]: {follow_up_question}"
            answer += f"\n[Follow-up Answer]: {follow_up_answer}"

        interview_transcript.append({"type": q_type, "question": question, "answer": answer})
        print("Bot: Thank you for that response.")

    with open("interview_transcript.json", "w") as f:
        json.dump(interview_transcript, f, indent=4)

    print("\n--- Interview Concluded ---")
    print("Bot: That's all the questions I have for you. Thank you for attending the Interview.")
    print("Bot: We will be in touch regarding the interview.")
    print("\nInterview transcript has been saved to 'interview_transcript.json'")

if __name__ == "__main__":
    run_interview_session()