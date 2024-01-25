from langchain.llms import LlamaCpp
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import json
import streamlit as st
import os

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
from langchain.schema import LLMResult


def csv_tool_openai(filename: str):
    df = pd.read_csv(filename)
    model = ChatOpenAI(
        temperature=0,
        model="gpt-4-1106-preview",
    )
    return create_pandas_dataframe_agent(
        model,
        df,
        verbose=True,
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
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
           {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
           {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Note: We only accommodate two types of charts: "bar" and "line".

        4. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

        REMENBER BE EXTREMEMLY CAREFUL WITH THE FORMATTING. ANY MISTAKE WILL CAUSE THE MODEL TO FAIL.
        if you don't use double quotes, the model will fail. Be extremaly careful with the formatting.

        Also remenber the tool is python_repl_ast, do not add backticks or any other character to the response.

        You're a 10x data analysts, you help me make an analysis of the dataset provided.

        Now, let's tackle the query step by step. Here's the query for you to work on: 
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


# st.set_page_config(page_title="Talk with your data")
# st.title("Talk with your data")

# st.write("Please upload your CSV file below.")

# data = st.file_uploader("Upload a CSV", type="csv")

# query = st.text_area("Send a Message")

# if st.button("Submit Query", type="primary"):
#     # Create an agent from the CSV file.
#     agent = csv_tool_openai(data)

#     # Query the agent.
#     response = ask_agent(agent=agent, query=query)

#     # Decode the response.
#     decoded_response = decode_response(response)

#     # Write the response to the Streamlit app.
#     write_answer(decoded_response)

# # #workflow romeo
# # #debug on
