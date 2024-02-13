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
st.set_page_config(page_title="Enhanced CSV Tool with LLM", layout='wide')
st.title("Enhanced CSV Tool with LLM")

# Upload CSV files for comparison
col0, col1 = st.columns(2)
data0 = col0.file_uploader("Upload first contract", type="csv")
data1 = col1.file_uploader("Upload second contract", type="csv")

# Upload multiple CSV files for chat analysis
input_csvs = st.sidebar.file_uploader("Upload your CSV files for analysis", type=['csv'], accept_multiple_files=True)

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
    query = st.text_area("Send a Message for Contract Comparison")
    if st.button("Submit Query for Comparison"):
        agent = csv_tool_openai(data0, data1)
        with st.spinner("Comparing Contracts..."):
            response = agent.chat(query)
        st.write(response)

# Handling file uploads for chat analysis
if input_csvs:
    selected_file = st.selectbox("Select a CSV file for Chat Analysis", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data)

    input_text = st.text_area("Enter your query for Chat Analysis")
    if input_text and st.button("Chat with CSV"):
        result = chat_with_csv(data, input_text)
        st.success(result)
