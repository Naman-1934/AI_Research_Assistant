from langchain_text_splitters import RecursiveCharacterTextSplitter


# LLM can not read 50 to 100 pages so, Each chunk contains 1000 characters.
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = splitter.split_text(text)

    return chunks