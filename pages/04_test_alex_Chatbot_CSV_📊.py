from langchain.chat_models import ChatOpenAI
import pandas as pd
import streamlit as st

from pandasai import SmartDatalake


def csv_tool_openai(filename0: str, filename1: str):
    df0 = pd.read_csv(filename0)
    df1 = pd.read_csv(filename1)
    model = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
    )
    return SmartDatalake([df0, df1], config={"llm": model})


st.set_page_config(page_title="Compare contracts 📊📊📊")
st.title("Compare contracts 📊📊📊")

st.write("Please upload your CSV file below.")

col0, col1 = st.columns(2)
data0 = col0.file_uploader("Upload first contract", type="csv")
data1 = col1.file_uploader("Upload second contract", type="csv")

if data0 and data1:

    query = st.text_area("Send a Message")

    if st.button("Submit Query", type="primary"):
        # Create an agent from the CSV file.
        agent = csv_tool_openai(data0, data1)

        # Query the agent.
        with st.spinner("Thinking..."):
            response = agent.chat(query)

        # Decode the response.
        st.write(response)
