import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

CHROMA_PATH = "chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def build_retriever():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

def build_rag_chain():
    retriever = build_retriever()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY
    )

    # Rewrites the user's question into a standalone search query
    # using conversation history as context
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are reformulating user questions into standalone search queries for a RAG retrieval system.
        Today is {today}.
        Return only the search query, nothing else."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def rewrite_query(x):
        rewrite_chain = contextualize_prompt | llm | StrOutputParser()
        rewritten = rewrite_chain.invoke({
            "input": x["question"],
            "chat_history": x["chat_history"],
            "today": x["today"]
        })
        return rewritten

    # Answers the question using retrieved context
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer support assistant for PeakForm Gym.
        Today is {today}.
        Answer the member's question using the information provided in the context below.
        Be friendly, concise, and specific — include prices, times, or details where relevant.
        If the context genuinely does not contain the answer, say "I don't have that information — please contact us on 01509 334 200 or email hello@peakformgym.co.uk"
        Context:
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": (lambda x: rewrite_query(x)) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "today": lambda x: x["today"]
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return chain