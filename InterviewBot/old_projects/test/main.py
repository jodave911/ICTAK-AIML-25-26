import os
from dotenv import load_dotenv

# Core LCEL classes
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Google specific classes
from langchain_google_genai import ChatGoogleGenerativeAI

# Community-provided classes for document handling, storage, and LOCAL embeddings
from langchain_community.document_loaders import TextLoader
# --- CHANGE #1: Import the better text splitter ---
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
# --- CHANGE #2: Import the local embeddings model ---
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# Modern chain constructors
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# --- 1. Setup Environment ---
load_dotenv()

class InterviewChatbot:
    def __init__(self, job_description_path: str, resume_path: str):
        if not os.path.exists(job_description_path) or not os.path.exists(resume_path):
            raise FileNotFoundError("Job description or resume file not found.")

        self.jd_path = job_description_path
        self.resume_path = resume_path
        self.full_context_docs = self._load_and_split_docs()
        self.conversation_chain = self._setup_rag_pipeline()
        self.chat_history = []

    def _load_and_split_docs(self):
        jd_loader = TextLoader(self.jd_path)
        resume_loader = TextLoader(self.resume_path)

        jd_docs = jd_loader.load()
        for doc in jd_docs: doc.metadata = {"source": "job_description"}

        resume_docs = resume_loader.load()
        for doc in resume_docs: doc.metadata = {"source": "resume"}

        documents = jd_docs + resume_docs
        # --- CHANGE #3: Use the more robust splitter ---
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        return text_splitter.split_documents(documents)

    def _setup_rag_pipeline(self):
        chunks = self.full_context_docs

        # --- CHANGE #4: Use a free, local embedding model to avoid API quota issues ---
        # The first time you run this, it will download the model (approx. 1.3 GB)
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="hkunlp/instructor-large",
            model_kwargs={"device": "cpu"} # Use "cuda" if you have a GPU
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # We still use Google's powerful model for the chat logic

        llm = HuggingFaceInstructEmbeddings(
            model_name="ibm-granite/granite-4.0-h-350m",
            model_kwargs={"device": "cpu"} # Use "cuda" if you have a GPU
        )


        # The rest of the chain setup is identical
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_system_prompt = (
            "You are an expert AI interviewer. Your goal is to conduct a screening interview. "
            "Use the provided context, which includes the job description and the candidate's resume, to ask relevant questions. "
            "Analyze the chat history to ask logical follow-up questions. Do not repeat questions. "
            "If the candidate's answer is short, probe for more details. If they go off-topic, gently guide them back. "
            "Your response should ONLY be the next question to ask the candidate.\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain

    # The start_interview and generate_report methods remain exactly the same.
    # I am omitting them here for brevity, but you should keep them in your file.
    def start_interview(self, num_questions=5):
        """
        Manages the interview flow. The logic is identical to the previous version.
        """
        print("--- AI Interview Initializing ---")
        print("Hello, I will be conducting your initial screening interview today. Let's begin.")

        user_input = "Based on my resume and the job description, please ask me the first question."

        for i in range(num_questions):
            result = self.conversation_chain.invoke({
                "chat_history": self.chat_history,
                "input": user_input
            })
            ai_question = result['answer']

            print(f"\nAI Interviewer: {ai_question}")

            candidate_answer = input("Your Answer: ")
            self.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=ai_question),
                HumanMessage(content=candidate_answer)
            ])

            user_input = "That was my answer. What is your next question?"

        print("\n--- Interview Concluded ---")
        print("Thank you for your time. We will be in touch regarding the next steps.")

    def generate_report(self):
        """
        Generates a detailed evaluation report for the hiring manager using Gemini.
        """
        print("\n--- Generating Evaluation Report ---")

        transcript = ""
        for i, msg in enumerate(self.chat_history):
            if isinstance(msg, AIMessage):
                transcript += f"Interviewer: {msg.content}\n"
            elif isinstance(msg, HumanMessage) and i > 0 and isinstance(self.chat_history[i-1], AIMessage):
                 transcript += f"Candidate: {msg.content}\n\n"

        jd_context = "\n".join([doc.page_content for doc in self.full_context_docs if doc.metadata['source'] == 'job_description'])
        resume_context = "\n".join([doc.page_content for doc in self.full_context_docs if doc.metadata['source'] == 'resume'])

        report_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Senior Hiring Manager. Your task is to analyze the provided interview transcript, job description, and candidate resume to write a comprehensive evaluation report. Your analysis must be objective and strictly based on the provided text."),
            ("human", """
            Please provide a detailed evaluation report based on the information below.

            **Job Description Context:**
            {jd}

            **Candidate Resume Context:**
            {resume}

            **Full Interview Transcript:**
            {transcript}

            ---
            **Instructions:**
            Generate the report in the following structured format:

            **1. Overall Summary:** A brief, two-sentence summary of the candidate's performance and suitability for the role.
            **2. Alignment with Job Description:**
               - **Strengths:** List 3-5 key strengths the candidate demonstrated that align directly with the job description's requirements.
               - **Weaknesses/Gaps:** Identify any potential weaknesses, gaps in knowledge, or areas where their responses did not fully meet the role's requirements.
            **3. Technical & Project-Specific Skills:**
               - **Evaluation:** Assess the candidate's explanation of their skills and projects. Did their answers demonstrate depth and confidence?
            **4. Communication Skills:**
               - **Clarity & Conciseness:** Rate their ability to communicate ideas clearly.
            **5. Red Flags (If any):** Note any concerning responses, inconsistencies, or evasive answers.
            **6. Final Recommendation:**
               - **[Proceed / Hold / Reject]**
               - **Justification:** Provide a clear reason for your recommendation.
            """)
        ])
        
        report_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, convert_system_message_to_human=True)
        report_chain = report_prompt_template | report_llm | StrOutputParser()

        final_report = report_chain.invoke({
            "jd": jd_context,
            "resume": resume_context,
            "transcript": transcript
        })

        print(final_report)
        return final_report

# --- Example Usage ---
if __name__ == "__main__":
   
    try:
        interview_bot = InterviewChatbot(
            job_description_path="jd_genai_engineer.txt",
            resume_path="resume_candidate_a.txt"
        )
        interview_bot.start_interview(num_questions=3)
        interview_bot.generate_report()

    except Exception as e:
        print(f"An error occurred: {e}")
    