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
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory

import tempfile

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Correctly setting the page configuration at the beginning of your script
st.set_page_config(page_title="Assistant chatbot")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if 'document' not in st.session_state:
    st.session_state.document = 'Aucun'

st.title("Assisstants Seyna 🤖")

st.image(
    Image.open("static/logo_seyna_150_orange.png"),
    width=300,
)

# Templates and Data: Define job templates and other necessary data structures.

jobs = {
    "Gestion de sinistre": ["Sinistre 1", "Sinistre 2", "Sinistre 3"],
    "Assistant sales": ["Assistant 1", "Assistant 2", "Assistant 3"],
}

sinistre_templates = {
    "Sinistre 1": [
        "Votre tâche est d'évaluer l'urgence du sinistre et de guider l'utilisateur dans les premières étapes.",
        "Vous devez identifier les documents importants que l'utilisateur doit fournir pour traiter le sinistre.",
        "Il est crucial de déterminer le contexte du sinistre pour anticiper les questions de l'assurance."
    ],
    "Sinistre 2": [
        "Expliquez les étapes de la procédure de réclamation après un sinistre.",
        "Fournissez des conseils pour accélérer le processus de réclamation.",
        "Déterminez si des tiers sont impliqués et comment cela affecte la réclamation."
    ],
    "Sinistre 3": [
        "Guidez l'utilisateur à travers les termes et conditions liés aux sinistres dans leur police.",
        "Assistez l'utilisateur dans l'estimation des dommages liés au sinistre.",
        "Informez l'utilisateur sur le suivi de leur dossier de sinistre et les délais de réponse."
    ]
}

sales_templates = {
    "Assistant 1": [
        "Aidez à identifier les produits d'assurance les plus pertinents pour les besoins de l'utilisateur.",
        "Expliquez les avantages de certaines offres d'assurance par rapport à d'autres.",
        "Fournissez des exemples de scénarios où les produits d'assurance pourraient être utiles."
    ],
    "Assistant 2": [
        "Donnez des arguments clés pour convaincre un utilisateur hésitant.",
        "Présentez des études de cas ou des témoignages de clients satisfaits.",
        "Recommandez des stratégies de vente basées sur le profil du client."
    ],
    "Assistant 3": [
        "Analysez le retour sur investissement des polices d'assurance pour le client.",
        "Présentez une comparaison des coûts des différentes options d'assurance.",
        "Éclairez sur la manière dont les changements de vie personnelle peuvent affecter les besoins en assurance."
    ]
}


# Helper functions

def prepare_file(uploaded_file):
    """
    Save an uploaded file to a temporary directory and return the file path.

    Args:
        uploaded_file: The uploaded file object from Streamlit.

    Returns:
        The file path where the uploaded file is saved.
    """
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path
    return None

def agent_rag(loader, template, doc_type="PDF"):
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    tool = create_retriever_tool(
        retriever,
        "search_in_document",
        "Searches and returns documents.",
    )
    tools = [tool]

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        openai_api_key=api_key,
    )

    context = (
        template
        + """

    Your task will be to complete the request of the user and using the provided {doc_type} by the user.If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise

    Remember it's very important your answer should always be in French

    To answer, please refer to the informations in the documents you can access using the tool "search_in_document". Never ask the user for the document. The document is already given. Use the function "search_in_document".
    """
    )
    sys_message = SystemMessage(content=context)

    agent_executor = create_conversational_retrieval_agent(
        llm,
        tools,
        system_message=sys_message,
        verbose=True,
    )

    return agent_executor

def query(agent, question):
    with st.spinner("Waiting for response..."):
        response = agent.invoke({"input": question})
        if "text" in response:
            response = response["text"]
        else:
            response = response["output"]
    return response

def update_agent(file, categorie, tache):
    """
    Create or update the agent based on the task and document type.

    Args:
        file: The uploaded file object.
        categorie: The selected category.
        tache: The selected task.

    Returns:
        An updated agent instance.
    """
    file_path = prepare_file(file)
    template = None

    if categorie == "Gestion de sinistre":
        template = sinistre_templates[tache][0]  # Selecting the first template for example
    elif categorie == "Assistant sales":
        template = sales_templates[tache][0]

    if not template:
        template = "Un template par défaut pour les cas non traités"  # Add a default template if necessary

    loader = None
    if file and categorie == "Gestion de sinistre" and tache == "Sinistre 1":
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith('.csv'):
            loader = CSVLoader(file_path)
        else:
            st.error("Format de fichier non pris en charge. Veuillez télécharger un fichier PDF ou CSV.")

    # Add other loader conditions for different categories and tasks if needed

    if loader:
        return agent_rag(loader, template)
    else:
        return None



# Streamlit app interface nd integrate helper functions above

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
file = st.file_uploader("Selectionnez le fichier", type=["pdf", "csv"])

# Choose category and task
categorie = st.selectbox("Choisissez votre categorie :", list(jobs.keys()))
tache = st.selectbox("Choisissez votre tache :", jobs[categorie])

# Update the agent if a file is uploaded
if file and categorie and tache:
    agent = update_agent(file, categorie, tache)
    if agent:
        st.session_state.agent = agent
    else:
        st.error("Erreur lors de la mise à jour de l'agent. Veuillez réessayer.")

# Display chat messages from history on app rerun
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input for questions and display the agent's response
response = ""
if "agent" in st.session_state and st.session_state.agent is not None:
    user_input = st.text_input("Posez votre question ici :")
    if user_input:
        response = query(st.session_state.agent, user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
