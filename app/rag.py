import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_PATH = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def build_rag_chain():
    """Set-up of RAG pipeline"""
    # Loads the vectorstore from ChomaDB
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful customer support assistant for PeakForm Gym.
    Answer the member's question using the information provided in the context below.
    Be friendly, concise, and specific — include prices, times, or details where relevant.
    If the context genuinely does not contain the answer, say "I don't have that information — please contact us on 01509 334 200 or email hello@peakformgym.co.uk"

    Context:
    {context}

    Question: {question}

    Answer:"""
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    chain = build_rag_chain()
    question = "What are the opening times?"
    result = chain.invoke(question) #execution of RAG pipeline
    print(result)
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    question = "how much is a membership"
    docs = vectorstore.similarity_search(question, k=4)
    
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content)