from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import yfinance
import json
import openai as client
import time
import streamlit as st


def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

assistant_id = "asst_rYyJpQ0Ti93nmwqvxtwX8c1y"


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


def wait_on_run(run_id, thread_id):
    run = get_run(run_id, thread_id)
    while run.status == "queued" or run.status == "in_progress":
        run = get_run(run_id, thread_id)
        time.sleep(0.5)
    return run


st.set_page_config(
    page_title="InvestorAssistant",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorAssistant

    Welcome to InvestorAssistant.
"""
)

message_content = st.text_input("")

if message_content:
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": message_content,
            }
        ]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    run = wait_on_run(run.id, thread.id)
    while run.status == "requires_action":
        run = submit_tool_outputs(run.id, thread.id)
        run = wait_on_run(run.id, thread.id)
    result = get_messages(thread.id)[-1].content[0].text.value
    st.write(result.replace("$", "\$"))