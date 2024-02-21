from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image

from langchain.schema.messages import SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

import tempfile

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title="Assistant Gestion de Sinistre Seyna")

# Style
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title and image
st.title("Assistant Gestion de Sinistre Seyna 🤖")
st.image(Image.open("static/logo_seyna_150_orange.png"), width=300)

# Initialize session state
if 'document' not in st.session_state:
    st.session_state.document = 'Aucun'

if "messages" not in st.session_state:
    st.session_state.messages = []

# Jobs for "Gestion de sinistre"
jobs = ["Sinistre 1", "Sinistre 2", "Sinistre 3"]

# Propositions for each "Gestion de sinistre" task
sinistre_templates = {
    "Sinistre 1": "...",  # Your task descriptions
    "Sinistre 2": "...",
    "Sinistre 3": "..."
}

# Task selection
tache = st.radio("Choisissez votre tache :", jobs)

# Document type selection
document_type = st.radio("Type de document :", ("Aucun", "PDF", "CSV"))

# File uploader
file = None
if document_type == "PDF":
    file = st.file_uploader("Selectionnez le pdf", type="pdf")
elif document_type == "CSV":
    file = st.file_uploader("Selectionnez le csv", type="csv")

# Helper functions
def prepare_file(uploaded_file):
    # Your existing code to handle the file
    ...

def agent_rag(loader, template):
    # Your existing code to create the agent
    ...

# Update agent based on task
if file:
    file_path = prepare_file(file)
    template = sinistre_templates[tache]
    loader = PyPDFLoader(file_path) if document_type == "PDF" else CSVLoader(file_path)
    st.session_state.agent = agent_rag(loader, template)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and agent response
if "agent" in st.session_state and st.session_state.agent is not None:
    user_input = st.text_input("Posez votre question ici :")
    if user_input:
        response = query(st.session_state.agent, user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)
