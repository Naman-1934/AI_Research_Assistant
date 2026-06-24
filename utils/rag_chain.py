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
        google_api_key = api_key    
    )

    return llm

def generate_answer(llm, context, question, chat_history):

    history_text = ""
    
    for msg in chat_history:

        role = msg['role'].upper()

        history_text += (f"{role}: {msg['content']}\n")

    prompt = f"""
You are an expert research assistant

Use only the provided context to answer the question.

If the answer is not in the context, say "I could not find this in the uploaded documents."

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
    prompt = f"""
You are an expert research analyst.

Create a detailed summary of the document.

Include:

1. Main Topic
2. Key Findigs
3. Important Concepts
4. Methodology (if applicable)
5. Limitations (if applicable)
6. Future Scope (if applicable)

Documents:

{text}

Summary:
"""

    response = llm.invoke(prompt)

    return response.content
