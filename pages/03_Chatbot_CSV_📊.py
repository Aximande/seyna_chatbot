from langchain.llms import LlamaCpp
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
from dotenv import load_dotenv
import json
import streamlit as st
import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.schema import LLMResult


class ChatBotCallbackHandler(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs: any) -> any:
        print("on_llm_end---------------------")
        print(response)


def csv_tool_openai(filename0: str, filename1: str):
    df0 = pd.read_csv(filename0)
    df1 = pd.read_csv(filename1)
    model = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
    )
    return create_pandas_dataframe_agent(
        model,
        [df0, df1],
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )


def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
    prompt = (
prompt = (
    """
    Let's decode the way to respond to the queries for our health insurance data. The responses depend on the type of information requested in the query.

    1. If the query requires information about reimbursements or out-of-pocket expenses, format your answer like this:
       {"table": {"columns": ["Category", "Type of Expense", "Reimbursement", "Amount Remaining"], "data": [[category1, expense1, reimbursement1, remaining1], [category2, expense2, reimbursement2, remaining2], ...]}}

    2. For a bar chart showing the comparison of reimbursements, use this format:
       {"bar": {"columns": ["Type of Expense", "Reimbursement", "Amount Remaining"], "data": [reimbursement1, remaining1, reimbursement2, remaining2, ...]}}

    3. If a line chart is needed to visualize the trend of reimbursements over time or for different categories, your reply should look like this:
       {"line": {"columns": ["Type of Expense", "Reimbursement", "Amount Remaining"], "data": [reimbursement1, remaining1, reimbursement2, remaining2, ...]}}

    Note: We only accommodate two types of charts: "bar" and "line".

    4. For a plain question that doesn't need a chart or table, your response should be straightforward:
       {"answer": "The answer to your question is here"}

    For example:
       {"answer": "The maximum reimbursement for a hospital stay is €650"}

    5. If the answer is not known or available, respond with:
       {"answer": "I do not have that information."}

    Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes.
    For example: {"columns": ["Category", "Type of Expense", "Reimbursement", "Amount Remaining"], "data": [["Hospitalisation", "Daily hospital fee", "€20.00", "€0.00"], ["Hospitalisation", "Private room per night", "€70.00", "€0.00"]]}
    
    Now, let's address the specific query you have regarding our health insurance product offerings. Here's the query for you to work on: 
    """
        + query
    )

    # Run the prompt through the agent and capture the response.
    response = agent.run(prompt)

    # Return the response converted to a string.
    return response.__str__()


def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    response = response[response.find("{") : response.rfind("}") + 1]
    return json.loads(response)


def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


st.set_page_config(page_title="Talk with your data")
st.title("Talk with your data")

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
            response = ask_agent(agent=agent, query=query)

        # Decode the response.
        decoded_response = decode_response(response)

        # Write the response to the Streamlit app.
        write_answer(decoded_response)
