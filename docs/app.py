import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_community.tools import DuckDuckGoSearchRun
from fastembed import TextEmbedding
from typing import List
import os

load_dotenv()

# -----------------------------------------------
# Proper Embeddings - BAAI/bge-small-en-v1.5
# -----------------------------------------------
class SimpleEmbeddings(Embeddings):
    def __init__(self):
        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def embed_documents(
        self, texts: List[str]
    ) -> List[List[float]]:
        texts = [str(t).strip() for t in texts if t]
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        text = str(text).strip()
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

# -----------------------------------------------
# FUNCTION 1: Extract text from PDF
# -----------------------------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# -----------------------------------------------
# FUNCTION 2: Split text into chunks
# -----------------------------------------------
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

# -----------------------------------------------
# FUNCTION 3: Create Vector Store
# -----------------------------------------------
def get_vector_store(text_chunks):
    text_chunks = [t for t in text_chunks if t.strip()]
    embeddings = SimpleEmbeddings()
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vector_store

# -----------------------------------------------
# FUNCTION 4: Web Search Tool
# -----------------------------------------------
search = DuckDuckGoSearchRun()

def web_search(query: str) -> str:
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"Search failed: {str(e)}"

# -----------------------------------------------
# FUNCTION 5: Query Router
# -----------------------------------------------
def route_query(question: str) -> str:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query router.
        Decide if the question should be answered from:
        1. 'vectorstore' - if question is about uploaded PDF
        2. 'websearch' - if question needs current/general info

        Reply with ONLY one word: 'vectorstore' or 'websearch'
        """),
        ("human", "{question}")
    ])

    chain = router_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    return result.strip().lower()

# -----------------------------------------------
# FUNCTION 6: Self Reflection
# -----------------------------------------------
def reflect_answer(question: str, answer: str) -> str:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    reflect_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an answer quality checker.
        Check if the answer properly addresses the question.
        If answer is good reply: 'good'
        If answer needs improvement reply: 'improve'
        Reply ONLY with one word.
        """),
        ("human", "Question: {question}\nAnswer: {answer}")
    ])

    chain = reflect_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "question": question,
        "answer": answer
    })
    return result.strip().lower()

# -----------------------------------------------
# FUNCTION 7: RAG Chain
# -----------------------------------------------
def get_rag_answer(
    question: str,
    vector_store,
    chat_history: list
) -> str:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
        Answer based on the context below only.
        If you don't know, say you don't know.

        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

# -----------------------------------------------
# FUNCTION 8: Agentic RAG - Main Brain
# -----------------------------------------------
def agentic_rag(
    question: str,
    vector_store,
    chat_history: list
) -> dict:

    steps = []

    # Step 1: Route Query
    steps.append("🔍 Routing query...")
    route = route_query(question)
    steps.append(f"📍 Routed to: **{route}**")

    # Step 2: Get Answer
    if route == "vectorstore":
        steps.append("📄 Searching PDF document...")
        answer = get_rag_answer(
            question,
            vector_store,
            chat_history
        )
    else:
        steps.append("🌐 Searching the web...")
        answer = web_search(question)

    # Step 3: Self Reflection
    steps.append("🤔 Checking answer quality...")
    quality = reflect_answer(question, answer)
    steps.append(f"✅ Quality check: **{quality}**")

    # Step 4: Improve if needed
    if quality == "improve":
        steps.append("🔄 Improving answer...")
        if route == "vectorstore":
            web_result = web_search(question)
            answer = answer + \
                "\n\nAdditional Info:\n" + web_result
        else:
            rag_result = get_rag_answer(
                question,
                vector_store,
                chat_history
            )
            answer = answer + \
                "\n\nFrom Document:\n" + rag_result

    steps.append("✨ Final answer ready!")

    return {
        "answer": answer,
        "steps": steps,
        "route": route
    }

# -----------------------------------------------
# Handle questions
# -----------------------------------------------
def handle_question(user_question):
    with st.spinner("Agent is thinking..."):
        result = agentic_rag(
            user_question,
            st.session_state.vector_store,
            st.session_state.chat_history
        )

    # Show agent steps
    with st.expander("🤖 Agent Steps"):
        for step in result["steps"]:
            st.write(step)

    # Update chat history
    st.session_state.chat_history.extend([
        HumanMessage(content=user_question),
        AIMessage(content=result["answer"])
    ])

    # Display chat
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(
                message.content
            )

# -----------------------------------------------
# Main App
# -----------------------------------------------
def main():
    st.set_page_config(
        page_title="Agentic RAG Chatbot",
        page_icon="🤖"
    )
    st.header("🤖 Agentic RAG with LLaMA 3")
    st.caption(
        "Powered by LangChain + Groq + FAISS"
    )

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Ask anything...")
    if user_question:
        if st.session_state.vector_store is None:
            st.warning("⚠️ Please upload a PDF first!")
        else:
            handle_question(user_question)

    with st.sidebar:
        st.subheader("📁 Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs here",
            accept_multiple_files=True,
            type="pdf"
        )
        if st.button("Process PDFs 🚀"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = \
                        get_vector_store(text_chunks)
                    st.success("✅ Done! Start chatting!")
            else:
                st.warning("⚠️ Upload at least one PDF!")

        st.divider()
        st.subheader("ℹ️ How It Works")
        st.write("1. 📄 Upload your PDF")
        st.write("2. 🔍 Agent routes your question")
        st.write("3. 📚 Searches PDF or Web")
        st.write("4. 🤔 Reflects on answer quality")
        st.write("5. ✨ Gives best answer!")

if __name__ == "__main__":
    main()