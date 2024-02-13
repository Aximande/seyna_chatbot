from langchain.chat_models import ChatOpenAI
import pandas as pd
import streamlit as st
from pandasai import SmartDatalake, SmartDataframe
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Setting up API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title="Comparateur de Garanties: Analyse Interactive sur multiples CSVs", layout='wide')
st.title("Comparateur de Garanties Santé 📊")
st.markdown("""
**Bienvenue sur notre Comparateur de Garanties **, votre outil privilégié pour l'analyse interactive de vos tableaux de garanties au format CSV. Le Comparateur vous permet notamment de comparer deux tableaux de remboursement pur un produit d'assurance Santé et fournit des aperçus détaillés des remboursements pour différents actes médicaux
""")


# Upload CSV files for comparison
col0, col1 = st.columns(2)
data0 = col0.file_uploader("Téléverser votre premier tableau d'exemple de remboursement au format CSV", type="csv")
data1 = col1.file_uploader("Téléverser votre deuxième tableau d'exemple de remboursement au format CSV", type="csv")

# Upload multiple CSV files for chat analysis
input_csvs = st.sidebar.file_uploader("Téléchargez tous vos fichiers CSV à analyser", type=['csv'], accept_multiple_files=True)

# Function to handle OpenAI chat
def chat_with_csv(df, prompt):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    result = pandas_ai.chat(prompt)
    return result

# Function to compare contracts
def csv_tool_openai(filename0: str, filename1: str):
    df0 = pd.read_csv(filename0)
    df1 = pd.read_csv(filename1)
    model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    return SmartDatalake([df0, df1], config={"llm": model})

# Handling file uploads for contract comparison
if data0 and data1:
    query = st.text_area("Envoyer un message pour la Comparaison des offres de remboursement.")
    if st.button("Soumettre une requête pour comparaison"):
        agent = csv_tool_openai(data0, data1)
        with st.spinner("Comparaison des offres en cours ...."):
            response = agent.chat(query)
        st.write(response)

# Handling file uploads for chat analysis
if input_csvs:
    selected_file = st.selectbox("Sélectionner un fichier CSV dans le menu déroulant pour commencer l'analyse avec le bot", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data)

    input_text = st.text_area("Saisissez votre requête ci-dessous pour commencer la conversation avec le bot.")
    if input_text and st.button("Chat avec le CSV"):
        result = chat_with_csv(data, input_text)
        st.success(result)
