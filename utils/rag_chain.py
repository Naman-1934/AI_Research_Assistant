import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():

    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0.3,
        google_api_key = os.getenv("GOOGLE_API_KEY")
    )

    return llm

def generate_answer(llm, context, question):
    prompt = """
You are an expert research assistant

Use only provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content
