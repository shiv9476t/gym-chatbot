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
        template="""You are a helpful customer support assistant for a gym.
        Use only the information provided below to answer the question.
        If the answer is not in the provided information, say "I don't have that information — please contact the gym directly."

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
    result = chain.invoke(question)
    print(result)