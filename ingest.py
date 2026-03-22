import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

DOCS_PATH = "data/gym_docs"
CHROMA_PATH = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class NoMergeTextSplitter(RecursiveCharacterTextSplitter):
    """Splits on separators but never merges small chunks back together.
    Used for structured documents like timetables where each section
    must remain its own chunk regardless of size."""
    def _merge_splits(self, splits, separator):
        return [s for s in splits if s.strip()]


# Default splitter for prose documents — merges chunks up to 500 chars
prose_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Section splitter for structured documents — never merges sections
section_splitter = NoMergeTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separators=["\n\n", "\n", " "]
)

# Map filename keywords to splitter
SPLITTER_MAP = {
    "class_schedule": section_splitter,
    "faq": section_splitter,
}

def get_splitter(source: str):
    for keyword, splitter in SPLITTER_MAP.items():
        if keyword in source:
            return splitter
    return prose_splitter


def load_documents():
    pdf_loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs = pdf_loader.load() + txt_loader.load()
    print(f"Loaded {len(docs)} document(s).")
    return docs


def chunk_documents(docs):
    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        splitter = get_splitter(source)
        doc_chunks = splitter.split_documents([doc])
        
        for chunk in doc_chunks:
            if "faq" in source:
                chunk.metadata["doc_type"] = "faq"
            elif "class_schedule" in source:
                chunk.metadata["doc_type"] = "schedule"
            elif "pricing" in source:
                chunk.metadata["doc_type"] = "pricing"
            elif "membership" in source:
                chunk.metadata["doc_type"] = "membership"
        
        chunks.extend(doc_chunks)
    
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def filter_chunks(chunks: list[Document]) -> list[Document]:
    def is_meaningful(chunk):
        source = chunk.metadata.get("source", "")
        min_length = 40 if "class_schedule" in source else 100
        return len(chunk.page_content.strip()) >= min_length

    filtered = [c for c in chunks if is_meaningful(c)]
    print(f"Filtered to {len(filtered)} chunks (removed {len(chunks) - len(filtered)} noise chunks).")
    return filtered


def store_embeddings(chunks):
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)

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
    filtered = filter_chunks(chunks)
    for i, chunk in enumerate(filtered):
        print(f"\n--- Chunk {i+1} (length: {len(chunk.page_content)}) ---")
        print(chunk.page_content)
    store_embeddings(filtered)
    print("Ingestion complete.")