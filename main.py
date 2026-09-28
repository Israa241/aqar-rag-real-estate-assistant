import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROK_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

import streamlit as st
st.title("Aqar عقار")
st.write("Your Trusted Real Estate Partne شريكك الموثوق في العقارات")

# 1. load document
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("aqar_real_estate_guide.pdf")
docs = loader.load()

# 2 splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter("chunk_size=100, chunk_overlap=5")
splits = text_splitter.split_documents(docs)

# 3. embedding - vector store - retriever
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",
    GEMINI_API_KEY = gemini_api_key
)
vectorstore = Chroma.from_documents(documents = splits, embedding = embeddings)
retriever = vectorstore.as_retriever()

# chat
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant for 'Aqar' real state company"),
    HumanMessagePromptTemplate.from_template("""
        context : {context},

        question : {question}
    """)
])
llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    api_key = groq_api_key
)
parser = StrOutputParser()
chain = ( {
    "context" : retriever,
    "question" : RunnablePassthrough()

} |prompt_template | llm | parser)


# connect streamlit
if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

user_input = st.chat_input("Say something...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role" : "user", "content": user_input})

    with st.chat_message("ai"):
        streamed_response = st.write_stream(chain.stream(user_input))
    st.session_state.messages.append({"role": "ai", "content": streamed_response})