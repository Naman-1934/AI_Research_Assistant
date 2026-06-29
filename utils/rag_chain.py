import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()

def get_llm():

    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    genai.configure(api_key=api_key)

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
        return(f"Error generating answer: {str(e)}")

def generate_summary(llm, text):
    prompt = f"""
You are an expert research analyst.

Create a detailed summary of the document.

Include:

1. Title (if available)
2. Objective
3. Problem Statement
4. Methodology
5. Dataset (if mentioned)
6. Proposed Model/Approach
7. Key Findings
8. Results
9. Advantages
10. Limitations
11. Future Scope
12. Conclusion

if any section is not available in the document, write:
"not mentioned in the paper".

Research Paper:
{text}

Summary:
"""

    response = llm.invoke(prompt)

    return response.content
