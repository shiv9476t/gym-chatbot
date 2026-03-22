
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
DOCS_PATH = "data/gym_docs"


class NoMergeTextSplitter(RecursiveCharacterTextSplitter):
    def _merge_splits(self, splits, separator):
        return [s for s in splits if s.strip()]

def load_documents():
    """Load all PDFs and txt files from the gym docs folder."""
    pdf_loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=TextLoader)

    docs = pdf_loader.load() + txt_loader.load()
    print(f"Loaded {len(docs)} document(s).")
    return docs

def chunk_documents(docs):
    """Split documents into chunks for embedding."""
    splitter = NoMergeTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


docs = load_documents()
chunks = chunk_documents(docs)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} (length: {len(chunk.page_content)}) ---")
    print(chunk.page_content)