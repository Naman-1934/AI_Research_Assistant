import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()


def get_api_key():
    """
    Get Gemini API key.

    Local Development  -> .env
    Streamlit Cloud    -> Secrets
    """

    # Try Streamlit Secrets first
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

    # Otherwise use .env
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please configure it in .env (local) or Streamlit Secrets (deployment)."
        )

    return api_key

def get_llm():
    """
    Load Gemini API key.

    Local:
        Uses .env

    Streamlit Cloud:
        Uses st.secrets
    """

    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Add it to your .env file (local) or Streamlit Secrets (deployment)."
        )

    genai.configure(api_key=api_key)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    return llm

def generate_answer(llm, context, question, chat_history=None):

    """
    Generate answer using Gemini
    and retrieved document context.
    """

    try:

        model = genai.GenerativeModel("gemini-2.5-flash")

        history_text = ""

        if chat_history:
            history_text = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in chat_history[-5:]
                ]
            )
    

        prompt = f"""
You are an expert AI Research Assistant.

Rules:

1. Answer ONLY from provided context.

2. If information is not available,
   say:
   "The uploaded documents do not containn enough information."

3. Never make assumptions.

4. Give clear and professional responses.

5. When possible:
   - Use bullet points
   - Use sections
   - Keep answers concise

Conversation History:
{history_text}

Document Context:
{context}

Question:
{question}

Answer:
"""

        response = model.generate_content(prompt)

        return response.text
    
    except Exception as e:
        error = str(e)

        if "RESOURCE_EXHAUSTED" in error or "429" in error:
            return (
                "⚠️ **AI Service Temporarily Unavailable**\n\n"
                "The Gemini API has reached its free usage limit.\n\n"
                "Please try again after a few minutes."
            )

        return (
            "❌ An unexpected error occurred while generating the answer.\n"
            "Please try again later."
        )

def generate_summary(llm, text):

    prompt = f"""
You are an expert research analyst.

Create a detailed summary of the document.

Include:

1. Title
2. Objective
3. Problem Statement
4. Methodology
5. Dataset
6. Proposed Model
7. Key Findings
8. Results
9. Advantages
10. Limitations
11. Future Scope
12. Conclusion

If any section is missing, write:
"Not mentioned in the paper."

Research Paper:

{text}

Summary:
"""

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        error = str(e)

        if "RESOURCE_EXHAUSTED" in error or "429" in error:

            return (
                "⚠️ **AI Service Temporarily Unavailable**\n\n"
                "The AI service has temporarily reached its usage limit.\n\n"
                "Please try again after a few minutes.\n\n"
                "Thank you for your patience."
            )

        return (
            "❌ Unable to generate the summary at the moment.\n"
            "Please try again later."
        )
