import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not found." 
        )

    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0.3,
        google_api_key = os.getenv("GOOGLE_API_KEY")
    )

    return llm

def generate_answer(llm, context, question, chat_history):

    history_text = ""
    
    for msg in chat_history:

        role = msg['role'].upper()

        history_text += (f"{role}: {msg['content']}\n")

    prompt = f"""
You are an expert research assistant

Use only provided context.

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content

def generate_summary(llm, text):
    prompt = """
You are an expert research analyst.

Create a detailed summary of the document.

Include:

1. Main Topic
2. Key Findigs
3. Important Concepts
4. Methodology
5. Limitations
6. Future Scope

Documents:

{text}

Summary:
"""

    response = llm.invoke(prompt)

    return response.content
