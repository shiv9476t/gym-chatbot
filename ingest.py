import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# --- Config ---
DOCS_PATH = "data/gym_docs"
CHROMA_PATH = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_documents():
    """Load all PDFs and txt files from the gym docs folder."""
    pdf_loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)

    docs = pdf_loader.load() + txt_loader.load()
    print(f"Loaded {len(docs)} document(s).")
    return docs

def chunk_documents(docs):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def store_embeddings(chunks):
    """Generate embeddings and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_PATH}'.")

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    store_embeddings(chunks)
    print("Ingestion complete.")