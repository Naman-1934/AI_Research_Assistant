import streamlit as st


def validators_api_key(api_key):
    """
    Validate Gemini API Key.
    """

    if not api_key:
        st.error(
            "Please enter your Gemini API Key."
        )
        return False
    return True


def validate_uploaded_files(uploaded_file):
    """
    Validate the uploaded PDF file.
    Returns True if a PDF has been uploaded,
    otherwise False.
    """

    if uploaded_file is None:
        st.warning("Please upload a research paper (PDF).")
        return False

    return True

def validate_question(question):
    """
    Validate user question.
    """

    if question is None:
        return False
    
    if len(question.strip()) == 0:
        st.warning(
            "Please enter a valid question"
        )
        return False
    return True

def validate_vector_store(vector_store):
    """
    Validate vector store existance.
    """

    if vector_store is None:
        st.warning(
            "Please process PDFs first."
        )
        return False
    return True

