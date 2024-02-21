from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from dotenv import load_dotenv
import json
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

# The very first Streamlit command
st.set_page_config(page_title="Assistant chatbot", layout="wide")
st.title("Assisstants Seyna 🤖")
# Assuming style.css is in the same directory as your script
css_file_path = os.path.join(os.path.dirname(__file__), "style.css")

if os.path.exists(css_file_path):
    with open(css_file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error("File 'style.css' not found.")

jobs = {
    "Gestion de sinistre": ["Sinistre 1", "Sinistre 2", "Sinistre 3"],
    "Assistant sales": ["Assistant 1", "Assistant 2", "Assistant 3"],
}

# Trois propositions pour chaque tâche de "Gestion de sinistre"
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

# Trois propositions pour chaque tâche de "Assistant sales"
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


def prepare_file(uploaded_file):
    path = None
    if uploaded_file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return path


def agent_without_rag(template):
    # LLM

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
        openai_api_key=api_key,
    )

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(template),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    return conversation


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
    # This combines both 'update_agent' definitions into one
    if not file:
        return None

    file_path = prepare_file(file)
    template = None

    if categorie == "Gestion de sinistre":
        template = sinistre_templates[tache][0]  # Selecting the first template for example
    elif categorie == "Assistant sales":
        template = sales_templates[tache][0]

    if not template:
        template = "Un template par défaut pour les cas non traités"  # Add a default template if necessary

    loader = PyPDFLoader(file_path)
    return agent_rag(loader, template, "PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.previous_agent = {
        "category": "Gestion de sinistre",
        "job": "Sinistre 1",
        "document": "Aucun",
    }


with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.image(
    Image.open("static/logo_seyna_150_orange.png"),
    width=300,
)
st.title("Chatbot 🤖")

st.radio(
    "Choisissez votre categorie :",
    (
        "Gestion de sinistre",
        "Assistant sales",
    ),
    key="categorie",
)

# we create a placeholder list to store the jobs in the category

st.radio(
    "Choisissez votre tache :",
    jobs[st.session_state.categorie],
    key="tache",
)

st.radio(
    "Document ?",
    ("Aucun", "PDF", "CSV"),
    key="document",
)

if st.session_state.document == "PDF":
    file = st.file_uploader("Selectionnez le pdf", type="pdf")
elif st.session_state.document == "CSV":
    file = st.file_uploader("Selectionnez le csv", type="csv")
else:
    file = None


# Initialize chat history and other session state variables if needed
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.previous_agent = {
        "category": None,
        "job": None,
        "document": None
    }

# Set up your Streamlit widgets here
st.title("Chatbot 🤖")

# Choose your category
categorie = st.radio("Choisissez votre categorie :", ("Gestion de sinistre", "Assistant sales"))

# Choose your job based on the category
tache = st.radio("Choisissez votre tache :", jobs[categorie])

# Choose the type of document
document_type = st.radio("Document ?", ("Aucun", "PDF", "CSV"))

# File uploader for PDF
if document_type == "PDF":
    pdf_file = st.file_uploader("Selectionnez le pdf", type="pdf")
    if pdf_file:
        st.session_state.agent = update_agent(pdf_file, categorie, tache)


elif document_type == "CSV":
    csv_file = st.file_uploader("Sélectionnez le CSV", type="csv")
    if csv_file:
        csv_path = prepare_file(csv_file)
        loader = CSVLoader(csv_path)

        # Select the correct template set based on the category
        if categorie == "Gestion de sinistre":
            template = sinistre_templates[tache][0]  # Select the first template for the task
        elif categorie == "Assistant sales":
            template = sales_templates[tache][0]  # Select the first template for the task
        else:
            template = "Un template par défaut pour les cas non traités."

        agent = agent_rag(loader, template, "CSV")
        st.session_state.agent = agent

        # Allow the user to interact with the CSV data
        user_question_csv = st.text_input("Posez une question concernant les données du CSV :")
        if user_question_csv:
            # Invoke the agent with the user's question
            response_csv = agent.invoke({"input": user_question_csv})
            # Store the response in session state
            st.session_state["csv_response"] = response_csv["text"] if "text" in response_csv else response_csv["output"]
            # Display the response
            st.write(st.session_state["csv_response"])


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
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display the agent's response
if response:
    st.markdown(response)
