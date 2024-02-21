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

    tool = create_retriever_tool(retriever, "search_in_document", "Searches and returns documents.")

    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", openai_api_key=api_key)
    sys_message = SystemMessage(content="En tant qu'assistant IA spécialisé, j'ai pour mission d'analyser avec précision les bulletins d'adhésion aux assurances santé. Mon objectif est d'extraire des données clés avec une attention particulière aux détails suivants : informations personnelles de l'adhérent, détails de la couverture d'assurance, informations sur les bénéficiaires, modalités de paiement, et toutes déclarations significatives. Pour chaque document fourni, je vais scrupuleusement identifier et résumer ces informations, en veillant à respecter la confidentialité et l'intégrité des données. Je suis programmé pour comprendre le contexte complexe des assurances santé et pour répondre de manière intelligente et contextuelle, en fournissant des informations exploitables et pertinentes basées exclusivement sur le contenu du document chargé. Votre demande sera traitée avec la plus grande rigueur pour assurer une extraction d'informations à la fois complète et conforme aux normes du secteur.")

    # Configuration correcte de l'agent_executor
    agent_executor = create_conversational_retrieval_agent(llm, [tool], system_message=sys_message, verbose=True)

    return agent_executor

def display_extraction_results(agent, extraction_query):
    """Affiche les résultats d'extraction automatique."""
    response = agent.invoke({"input": extraction_query})
    st.write(response["text"] if "text" in response else response["output"])

st.set_page_config(page_title="Assistant d'Extraction d'infos sur le Bulletin d'Adhésion (santé)")
st.title("Extractor Bulletin Adhésion")

st.markdown("""
**Bienvenue sur notre bot extracteur d'infos de bulletin d'adhésion**, votre outil privilégié pour extraire des informations clés des bulletins d'adhésion aux assurances santé.
""")


file = st.file_uploader("Téléchargez le bulletin d'adhésion (PDF)", type="pdf")

extraction_query = """
Sur la base du PDF de bulletin d'adhésion fourni, voici les champs à extraire pour un contexte assurantiel santé :

Informations personnelles :

Civilité
Nom
Prénom
Situation de famille
Date de naissance
Adresse complète (rue, code postal, ville)
Régime (par exemple, Sécurité sociale)
Coordonnées de contact :

Numéro de téléphone
Adresse e-mail
Détails de l'adhésion :

Numéro de demande d'adhésion
Date d'effet
Offres sélectionnées et niveaux de garantie
Cotisations (mensuelles, annuelles, totales)
Informations sur les ayants droit (si applicable)
Modalités de paiement :

Mode de paiement
Périodicité de paiement
Date de prélèvement
Déclarations et certifications :

Certifications concernant les Personnes Politiquement Exposées (PPE)
Consentement au traitement des données personnelles
Mentions légales et contacts :

Informations sur l'assureur, le courtier, et le gestionnaire de contrat
Informations sur la médiation et les réclamations
Signature et date :

Signature de l'adhérent
Date de signature
"""  # Votre question pré-définie ici

if file:
    file_path = prepare_file(file)
    agent = create_extraction_agent(file_path)
    display_extraction_results(agent, extraction_query)

    # Mise en forme améliorée pour la section Q&A
    st.markdown("## Posez une question supplémentaire concernant le bulletin d'adhésion")
    with st.container():  # Utilisez un conteneur pour la mise en page
        col1, col2 = st.columns([1, 4])  # Créez deux colonnes, la première plus étroite que la seconde
        with col1:
            st.write("Q:")  # 'Q:' dans la colonne plus étroite
        with col2:
            user_question = st.text_input("", key="user_question")  # Champ de saisie pour la question
            if user_question:
                response = agent.invoke({"input": user_question})
                st.session_state["user_response"] = response["text"] if "text" in response else ""
                st.markdown("### Réponse :")
                st.write(st.session_state["user_response"])  # Affichez la réponse de l'agent
