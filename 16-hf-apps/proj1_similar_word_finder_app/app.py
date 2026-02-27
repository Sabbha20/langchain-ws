import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Similar Word Finder", page_icon=":robot:")
st.header("Hey, Ask me something & I'll find similar words for you! :robot:")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
loader = CSVLoader(
    file_path="myData.csv",
    csv_args={
        'delimiter': ",",
        'quotechar': '"',
        'fieldnames': ['Words']
    }
)

data = loader.load()

db = FAISS.from_documents(data, embeddings)

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()
submit = st.button("Find Similar Things!!!")

if submit and user_input:
    docs = db.similarity_search(user_input, k=5)
    st.subheader("Top 5 Matches:")
    for i, doc in enumerate(docs[:5]):
        st.write(f"{i+1}. {doc.page_content}".replace("Words: ", ""))