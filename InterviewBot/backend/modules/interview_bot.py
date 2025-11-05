# import os
# import json
# import random
# import logging
# from typing import List, Dict, Tuple

# # LangChain components
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # --- 1. HELPER FUNCTIONS (Setup, Logging, Question Generation, Evaluation) ---

# def _setup_logging(log_file: str):
#     """Configures the logger to write LLM interactions to a specific file."""
#     # Remove any existing handlers to avoid duplicate logs
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)

#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         filename=log_file,
#         filemode='a'
#     )
#     print(f"Logging is configured. LLM interactions will be saved to '{log_file}'.")

# def _setup_environment(jd_index_path: str, resume_index_path: str):
#     """Initializes models and loads vector stores from specified paths."""
#     print("Initializing models and loading vector stores for interview...")
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     if not os.path.exists(jd_index_path) or not os.path.exists(resume_index_path):
#         raise FileNotFoundError("FAISS index directories not found. Please run the processing step first.")

#     jd_db = FAISS.load_local(jd_index_path, embeddings, allow_dangerous_deserialization=True)
#     resume_db = FAISS.load_local(resume_index_path, embeddings, allow_dangerous_deserialization=True)
#     print("Setup complete.")
#     return llm, jd_db, resume_db

# def _log_and_invoke_llm(chain, inputs: dict, purpose: str):
#     """Logs the request and response of an LLM invocation."""
#     logging.info(f"--- LLM Request Start ---")
#     logging.info(f"Purpose: {purpose}")
#     logging.info(f"Input: {json.dumps(inputs, indent=2)}")
#     response = chain.invoke(inputs)
#     response_content = response.content if hasattr(response, 'content') else str(response)
#     logging.info(f"Output: {response_content}")
#     logging.info(f"--- LLM Request End ---\n")
#     return response

# # (All four question generation functions: _generate_rag_question, etc.)
# def _generate_rag_question(llm: ChatGoogleGenerativeAI, jd_topic: Document, resume_db: FAISS) -> str:
#     resume_context_docs = resume_db.similarity_search(jd_topic.page_content, k=1)
#     resume_context = resume_context_docs[0].page_content if resume_context_docs else "No specific context found."
#     prompt = ChatPromptTemplate.from_template(
#         "You are an expert technical interviewer. Create a SINGLE, FOCUSED behavioral question that connects a job requirement to the candidate's resume."
#         "DO NOT ask multiple questions or use numbered lists.\n\nJOB REQUIREMENT: {jd_context}\nCANDIDATE'S RESUME SNIPPET: {resume_context}\n\nSingle Question (e.g., 'The role requires experience in X. I see on your resume you did Y. "
#         "Can you describe a specific challenge you faced in that project?'):"
#     )
#     chain = prompt | llm
#     inputs = {"jd_context": jd_topic.page_content, "resume_context": resume_context}
#     return _log_and_invoke_llm(chain, inputs, "Generate RAG Question").content

# def _generate_jd_question(llm: ChatGoogleGenerativeAI, jd_topic: Document) -> str:
#     prompt = ChatPromptTemplate.from_template(
#         "You are an expert interviewer. Based on the following job requirement, ask ONE concise question to gauge the candidate's understanding. The question must be a single sentence. Do not use lists.\n\nJOB REQUIREMENT: {jd_context}\n\nSingle Question:"
#     )
#     chain = prompt | llm
#     inputs = {"jd_context": jd_topic.page_content}
#     return _log_and_invoke_llm(chain, inputs, "Generate JD-focused Question").content

# def _generate_resume_question(llm: ChatGoogleGenerativeAI, resume_topic: Document) -> str:
#     prompt = ChatPromptTemplate.from_template(
#         "You are an expert interviewer. Based on this snippet from the candidate's resume, ask ONE detailed question to probe their experience. "
#         "The question must be a single, focused query.\n\nRESUME SNIPPET: {resume_context}\n\nSingle Question (e.g., 'Tell me more about your specific role in the X project mentioned here.'):"
#     )
#     chain = prompt | llm
#     inputs = {"resume_context": resume_topic.page_content}
#     return _log_and_invoke_llm(chain, inputs, "Generate Resume-focused Question").content

# def _generate_situational_question(llm: ChatGoogleGenerativeAI, jd_topic: Document) -> str:
#     prompt = ChatPromptTemplate.from_template(
#         "You are a hiring manager. Based on the following job responsibility, create ONE practical, hypothetical scenario question to test problem-solving skills."
#         " The output must be a single, concise question. DO NOT use bullet points or numbered lists.\n\nJOB RESPONSIBILITY: {jd_context}\n\nSingle Question (start with 'Imagine...' or 'Suppose...'):"
#     )
#     chain = prompt | llm
#     inputs = {"jd_context": jd_topic.page_content}
#     return _log_and_invoke_llm(chain, inputs, "Generate Situational Question").content

# def _evaluate_answer(llm: ChatGoogleGenerativeAI, question: str, answer: str) -> Tuple[str, str]:
#     """Evaluates the candidate's answer and decides if a follow-up is needed."""
#     prompt = ChatPromptTemplate.from_template(
#         "You are a senior interviewer evaluating a candidate's answer. Your goal is to assess if the answer has sufficient detail. A good answer provides specific examples. "
#         "A weak answer is vague or too short.\n\nOriginal Question: {question}\nCandidate's Answer: {answer}\n\nFirst, on a new line, write your decision: 'SUFFICIENT' or 'INSUFFICIENT'.\n"
#         "If 'INSUFFICIENT', on the next line, write a concise follow-up question to probe for more detail.\nIf 'SUFFICIENT', write 'None' on the second line.\n\nFormat:\nDecision: [SUFFICIENT/INSUFFICIENT]\nFollow-up: [Your follow-up question or None]"
#     )

#     chain = prompt | llm
#     inputs = {"question": question, "answer": answer}
#     response = _log_and_invoke_llm(chain, inputs, "Evaluate Candidate Answer")
#     try:
#         lines = response.content.strip().split('\n')
#         decision = lines[0].replace("Decision:", "").strip()
#         follow_up = lines[1].replace("Follow-up:", "").strip()
#         return decision, (None if follow_up.lower() == 'none' else follow_up)
#     except IndexError:
#         return "SUFFICIENT", None

# # --- 2. MAIN PUBLIC FUNCTION ---

# def run_interview_session(jd_index_path: str, resume_index_path: str, transcript_path: str, log_file: str):
#     """
#     Orchestrates the entire interactive interview process.
#     """
#     _setup_logging(log_file)
#     llm, jd_db, resume_db = _setup_environment(jd_index_path, resume_index_path)

#     while True:
#         try:
#             num_questions_str = input("Enter the number of questions to ask for the interview (e.g., 5): ")
#             num_questions = int(num_questions_str)
#             if num_questions > 0:
#                 break
#             else:
#                 print("Please enter a positive number.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")

#     jd_topics = jd_db.similarity_search("key responsibilities and qualifications", k=10)
#     resume_topics = resume_db.similarity_search("work experience and projects", k=10)

#     question_generators = {
#         "RAG": (_generate_rag_question, {"jd_topic": lambda: random.choice(jd_topics), "resume_db": resume_db}),
#         "SITUATIONAL": (_generate_situational_question, {"jd_topic": lambda: random.choice(jd_topics)}),
#         "JD": (_generate_jd_question, {"jd_topic": lambda: random.choice(jd_topics)}),
#         "RESUME": (_generate_resume_question, {"resume_topic": lambda: random.choice(resume_topics)})
#     }
#     question_types = list(question_generators.keys())
#     interview_transcript = []

#     print("\n--- Starting Interview ---")
#     print("Bot: Hello! Thank you for joining me today. I'd like to ask you a few questions.")

#     for i in range(num_questions):
#         q_type = random.choice(question_types)
#         generator_func, kwargs_template = question_generators[q_type]
#         kwargs = {key: value() if callable(value) else value for key, value in kwargs_template.items()}
#         kwargs["llm"] = llm
#         question = generator_func(**kwargs)

#         print(f"\nBot (Question {i+1}/{num_questions}, Type: {q_type}): {question}")
#         answer = input("You: ")

#         decision, follow_up_question = _evaluate_answer(llm, question, answer)
#         if decision == "INSUFFICIENT" and follow_up_question:
#             print(f"Bot: {follow_up_question}")
#             follow_up_answer = input("You: ")
#             question += f"\n[Follow-up Question]: {follow_up_question}"
#             answer += f"\n[Follow-up Answer]: {follow_up_answer}"

#         interview_transcript.append({"type": q_type, "question": question, "answer": answer})
#         print("Bot: Thank you for that response.")

#     with open(transcript_path, "w") as f:
#         json.dump(interview_transcript, f, indent=4)

#     print("\n--- Interview Concluded ---")
#     print("Bot: That's all the questions I have for you. Thank you for attending the Interview.")
#     print("Bot: We will be in touch regarding the interview.")
#     print(f"\nInterview transcript has been saved to '{transcript_path}'")



import os
import json
import random
import logging
from typing import List, Dict, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class InterviewBot:
    def __init__(self, jd_index_path: str, resume_index_path: str, log_file: str, num_questions: int):
        self.log_file = log_file
        self.num_questions = num_questions
        self._setup_logging()
        self.llm, self.jd_db, self.resume_db = self._setup_environment(jd_index_path, resume_index_path)

        self.jd_topics = self.jd_db.similarity_search("key responsibilities and qualifications", k=10)
        self.resume_topics = self.resume_db.similarity_search("work experience and projects", k=10)

        self.question_generators = {
            "RAG": (self._generate_rag_question, {"jd_topic": lambda: random.choice(self.jd_topics), "resume_db": self.resume_db}),
            "SITUATIONAL": (self._generate_situational_question, {"jd_topic": lambda: random.choice(self.jd_topics)}),
            "JD": (self._generate_jd_question, {"jd_topic": lambda: random.choice(self.jd_topics)}),
            "RESUME": (self._generate_resume_question, {"resume_topic": lambda: random.choice(self.resume_topics)})
        }
        self.question_types = list(self.question_generators.keys())
        self.interview_transcript = []
        self.current_question_details = {} # To hold the current question info

    # --- Private Helper Methods (Your original functions, now part of the class) ---

    def _setup_logging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=self.log_file, filemode='a')
        print(f"Logging is configured. LLM interactions will be saved to '{self.log_file}'.")

    def _setup_environment(self, jd_index_path: str, resume_index_path: str):
        print("Initializing models and loading vector stores for interview...")
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3) # Adjusted model for wider availability
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if not os.path.exists(jd_index_path) or not os.path.exists(resume_index_path):
            raise FileNotFoundError("FAISS index directories not found. Please run the processing step first.")
        jd_db = FAISS.load_local(jd_index_path, embeddings, allow_dangerous_deserialization=True)
        resume_db = FAISS.load_local(resume_index_path, embeddings, allow_dangerous_deserialization=True)
        print("Setup complete.")
        return llm, jd_db, resume_db

    def _log_and_invoke_llm(self, chain, inputs: dict, purpose: str):
        logging.info(f"--- LLM Request Start ---\nPurpose: {purpose}\nInput: {json.dumps(inputs, indent=2)}")
        response = chain.invoke(inputs)
        response_content = response.content if hasattr(response, 'content') else str(response)
        logging.info(f"Output: {response_content}\n--- LLM Request End ---\n")
        return response

    def _generate_rag_question(self, jd_topic: Document, resume_db: FAISS) -> str:
        resume_context_docs = resume_db.similarity_search(jd_topic.page_content, k=1)
        resume_context = resume_context_docs[0].page_content if resume_context_docs else "No specific context found."
        prompt = ChatPromptTemplate.from_template("You are an expert technical interviewer. Create a SINGLE, FOCUSED behavioral question that connects a job requirement to the candidate's resume. DO NOT ask multiple questions or use numbered lists.\n\nJOB REQUIREMENT: {jd_context}\nCANDIDATE'S RESUME SNIPPET: {resume_context}\n\nSingle Question (e.g., 'The role requires experience in X. I see on your resume you did Y. Can you describe a specific challenge you faced in that project?'):")
        chain = prompt | self.llm
        inputs = {"jd_context": jd_topic.page_content, "resume_context": resume_context}
        return self._log_and_invoke_llm(chain, inputs, "Generate RAG Question").content

    def _generate_jd_question(self, jd_topic: Document) -> str:
        prompt = ChatPromptTemplate.from_template("You are an expert interviewer. Based on the following job requirement, ask ONE concise question to gauge the candidate's understanding. The question must be a single sentence. Do not use lists.\n\nJOB REQUIREMENT: {jd_context}\n\nSingle Question:")
        chain = prompt | self.llm
        inputs = {"jd_context": jd_topic.page_content}
        return self._log_and_invoke_llm(chain, inputs, "Generate JD-focused Question").content

    def _generate_resume_question(self, resume_topic: Document) -> str:
        prompt = ChatPromptTemplate.from_template("You are an expert interviewer. Based on this snippet from the candidate's resume, ask ONE detailed question to probe their experience. The question must be a single, focused query.\n\nRESUME SNIPPET: {resume_context}\n\nSingle Question (e.g., 'Tell me more about your specific role in the X project mentioned here.'):")
        chain = prompt | self.llm
        inputs = {"resume_context": resume_topic.page_content}
        return self._log_and_invoke_llm(chain, inputs, "Generate Resume-focused Question").content

    def _generate_situational_question(self, jd_topic: Document) -> str:
        prompt = ChatPromptTemplate.from_template("You are a hiring manager. Based on the following job responsibility, create ONE practical, hypothetical scenario question to test problem-solving skills. The output must be a single, concise question. DO NOT use bullet points or numbered lists.\n\nJOB RESPONSIBILITY: {jd_context}\n\nSingle Question (start with 'Imagine...' or 'Suppose...'):")
        chain = prompt | self.llm
        inputs = {"jd_context": jd_topic.page_content}
        return self._log_and_invoke_llm(chain, inputs, "Generate Situational Question").content

    def _evaluate_answer(self, question: str, answer: str) -> Tuple[str, str]:
        prompt = ChatPromptTemplate.from_template("You are a senior interviewer evaluating a candidate's answer. Your goal is to assess if the answer has sufficient detail. A good answer provides specific examples. A weak answer is vague or too short.\n\nOriginal Question: {question}\nCandidate's Answer: {answer}\n\nFirst, on a new line, write your decision: 'SUFFICIENT' or 'INSUFFICIENT'.\nIf 'INSUFFICIENT', on the next line, write a concise follow-up question to probe for more detail.\nIf 'SUFFICIENT', write 'None' on the second line.\n\nFormat:\nDecision: [SUFFICIENT/INSUFFICIENT]\nFollow-up: [Your follow-up question or None]")
        chain = prompt | self.llm
        inputs = {"question": question, "answer": answer}
        response = self._log_and_invoke_llm(chain, inputs, "Evaluate Candidate Answer")
        try:
            lines = response.content.strip().split('\n')
            decision = lines[0].replace("Decision:", "").strip()
            follow_up = lines[1].replace("Follow-up:", "").strip()
            return decision, (None if follow_up.lower() == 'none' else follow_up)
        except IndexError:
            return "SUFFICIENT", None

    def _get_new_primary_question(self):
        q_type = random.choice(self.question_types)
        generator_func, kwargs_template = self.question_generators[q_type]
        kwargs = {key: value() if callable(value) else value for key, value in kwargs_template.items()}
        question = generator_func(**kwargs)
        self.current_question_details = {"type": q_type, "question": question, "answer": ""}
        return question

    # --- Public Methods for Streamlit Interaction ---

    def start_interview(self) -> str:
        """Generates the first question and returns it."""
        welcome_message = "Hello! Thank you for joining me today. I'd like to ask you a few questions. Let's start."
        first_question = self._get_new_primary_question()
        return f"{welcome_message}\n\n{first_question}"

    def process_user_answer(self, answer: str) -> str:
        """Processes the user's answer and returns the next question or a concluding message."""
        # Update the answer for the current question
        self.current_question_details["answer"] += answer

        decision, follow_up_question = self._evaluate_answer(
            self.current_question_details["question"],
            self.current_question_details["answer"]
        )

        if decision == "INSUFFICIENT" and follow_up_question:
            # Ask the follow-up question
            self.current_question_details["question"] += f"\n[Follow-up Question]: {follow_up_question}"
            self.current_question_details["answer"] += "\n[Follow-up Answer]: "
            return follow_up_question
        else:
            # The answer was sufficient, so log it and move to the next question
            self.interview_transcript.append(self.current_question_details)

            if len(self.interview_transcript) >= self.num_questions:
                return "END_OF_INTERVIEW"
            else:
                next_question = self._get_new_primary_question()
                return f"Thank you for that response.\n\n{next_question}"

    def save_transcript(self, transcript_path: str):
        """Saves the collected transcript to a file."""
        with open(transcript_path, "w") as f:
            json.dump(self.interview_transcript, f, indent=4)
        print(f"\nInterview transcript has been saved to '{transcript_path}'")