import streamlit as st
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import SystemMessage
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool

# Chargement des variables d'environnement
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def prepare_file(uploaded_file):
    """Prépare le fichier PDF téléchargé pour le traitement."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
    return tmp_file.name

def create_extraction_agent(file_path):
    """Crée un agent d'extraction basé sur le contenu du fichier PDF."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    tool = create_retriever_tool(retriever, "search_in_document", "Effectue des recherches et retourne des documents.")

    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", openai_api_key=api_key)
    sys_message = SystemMessage(content="En tant qu'assistant IA, je suis spécialisé dans l'analyse de baux de location pour en extraire des informations clés telles que les noms du bailleur et du locataire, les dates d'effet, les clauses de résiliation, les montants de loyer, charges, caution, etc. Je vais identifier et résumer ces informations de manière précise, tout en assurant la confidentialité et l'intégrité des données.")

    agent_executor = create_conversational_retrieval_agent(llm, [tool], system_message=sys_message, verbose=True)

    return agent_executor

def display_extraction_results(agent, extraction_query):
    """Affiche les résultats d'extraction automatique."""
    response = agent.invoke({"input": extraction_query})
    st.write(response["text"] if "text" in response else response["output"])

# Améliorations de l'interface utilisateur
st.set_page_config(page_title="Extracteur d'Infos de Bail", layout="wide")

st.title("📄 Extracteur d'Informations de Bail")

st.markdown("""
**Bienvenue sur l'Extracteur d'Informations de Bail**, votre outil privilégié pour extraire efficacement et précisément les détails clés des contrats de location.
""")

file = st.file_uploader("Téléchargez le contrat de location (PDF)", type="pdf", help="Seul le format PDF est pris en charge.")

extraction_query = """
Sur la base du PDF du contrat de location fourni, voici les champs à extraire :

**Identification des parties**
Nom :
adresse :
contact du bailleur :
contact du du locataire :

**Objet du contrat :**
Adresse et description détaillée du logement, y compris type d'habitat, surface habitable, nombre de pièces, et équipements.

**Conditions financières**
Montant du loyer (mensuel):
Contribution pour le partage des économies de charges:
Assurance récupérable pour le compte des colocataires:
Montant des charges récupérables :
modalités de paiement :
date ou période de paiement :
dépôt de garantie :
éventuelle contribution pour partage des économies de charges :

**Durée du contrat **
Date de prise d'effet :
durée :
conditions de renouvellement ou de résiliation.

**Clauses spécifiques**
Obligations et droits des parties :
conditions d'entretien et de réparation :
garanties :
modalités de résiliation :

**Annexes**
Diagnostic technique, état des lieux, autorisations préalables, et références aux loyers habituellement constatés.

"""  # Votre question pré-définie ici

if file:
    file_path = prepare_file(file)
    agent = create_extraction_agent(file_path)
    display_extraction_results(agent, extraction_query)

    st.header("Questions Supplémentaires sur le Contrat de Location")
    user_question = st.text_area("Posez une question supplémentaire concernant le contrat de location :", height=150)
    if user_question:
        response = agent.invoke({"input": user_question})
        st.session_state["user_response"] = response["text"] if "text" in response else response["output"]
        st.subheader("Réponse de l'assistant :")
        st.text_area("", value=st.session_state["user_response"], height=150, disabled=True)
