import os
from dotenv import load_dotenv

from langchain_ollama.llms import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# print(os.getenv("LANGCHAIN_API_KEY"))
# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT") or "olala_chat_app"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant please respond to the user queries."),
        ("human", "{question}")
        
    ]
)

# Streamlit app
st.title("Olala Gemma Chatapp")
input_text = st.text_input("What do you want to know? Go ahead and ask me - The OLALA Oracle!")

# ollama model
llm = OllamaLLM(model="gemma3:latest")
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

