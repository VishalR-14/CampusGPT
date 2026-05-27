from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_embeddings(text: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    vectorstore.persist()

    return len(chunks)