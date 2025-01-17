from typing import Optional, Type

import json
import traceback
import re

from model_configurations import get_model_configuration

import requests
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import Tool, StructuredTool, BaseTool
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

gpt_chat_version = "gpt-4o"
gpt_config = get_model_configuration(gpt_chat_version)


hw01_system_prompt = """
You are a calendar info provider, reply the info I asked, and reply in the requested format.
The format requirement:
- In language Traditional Chinese (zh-TW)
- Reply in JSON format.
- The JSON format is {format}
- Format the JSON output with indent of 4 spaces
- REPLY JSON ONLY. DO NOT REPLY WITH MARKDOWN CODE BLOCK SYNTEX.
"""
hw01_json_format = """
`{"Result":[{"date":"YYYY-MM-DD","name":"(name of the day)"}]}`
"""

hw02_system_prompt = """
Please reply my questions. Not only reply tool outputs.
Format requirement:
- Replay in Traditional Chinese (zh-TW)
- Reply in JSON format.
- The JSON format is `{"Result":[{"date":"YYYY-MM-DD","name":"(name of the day)"}]}`
- Format the JSON output with indent of 4 spaces
- REPLY JSON ONLY. DO NOT REPLY WITH MARKDOWN CODE BLOCK SYNTEX.
"""


def generate_hw01(question):
    llm = AzureChatOpenAI(
        model=gpt_config["model_name"],
        deployment_name=gpt_config["deployment_name"],
        openai_api_key=gpt_config["api_key"],
        openai_api_version=gpt_config["api_version"],
        azure_endpoint=gpt_config["api_base"],
        temperature=gpt_config["temperature"],
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", hw01_system_prompt), ("human", "{question}")]
    )
    content = llm.invoke(
        prompt.format(question=question, format=hw01_json_format)
    ).content
    return str(content)


def get_holiday_info_remote(year: str, month: str) -> str:
    """
    Request holiday info from remote API and make it a eazy format for LLM to take as input.
    return: json `{"Result":[{"date":"YYYY-MM-DD","name":"(name of the day)"}]}`
    """
    # key will be dispose once result is pass.
    api_key = "uWtaqONZuC6YAJ1bAHFgXHD3Y2qC0wNF"
    api_end_point = f"https://calendarific.com/api/v2/holidays?&api_key={api_key}&country=tw&year={year}&month={month}"
    api_end_point.format()
    json_responce = requests.get(api_end_point).json()
    result = []
    for holiday in json_responce["response"]["holidays"]:
        result.append({"name": holiday["name"], "date": holiday["date"]["iso"]})
    return json.dumps(result, indent=4)


def tool_get_holiday_info():
    def get_haliday_info(year: int, month: int) -> str:
        return get_holiday_info_remote(str(year), str(month))

    class GetHoliday(BaseModel):
        year: int = Field(description="specific year")
        month: int = Field(description="specific month")

    return StructuredTool.from_function(
        name="get_haliday_info",
        description="Query holidays for specific year and month",
        func=get_haliday_info,
        args_schema=GetHoliday,
    )


def generate_hw02(question):
    llm = AzureChatOpenAI(
        model=gpt_config["model_name"],
        deployment_name=gpt_config["deployment_name"],
        openai_api_key=gpt_config["api_key"],
        openai_api_version=gpt_config["api_version"],
        azure_endpoint=gpt_config["api_base"],
        temperature=gpt_config["temperature"],
    )
    tools = [tool_get_holiday_info()]
    # prompt mod from https://smith.langchain.com/hub/hwchase17/openai-functions-agent
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
            ("human", "{input}"),
        ]
    )
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    intput = hw02_system_prompt + "\n MY QUESTION IS: " + question
    return agent_executor.invoke({"input": intput})["output"]


def generate_hw03(question2, question3):
    pass


def generate_hw04(question):
    pass


def demo(question):
    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
        ]
    )
    response = llm.invoke([message])
    return response


if __name__ == "__main__":
    print("====main function start====")
    # print(generate_hw01("2024年台灣10月紀念日有哪些?"))
    # print(getHolidaysFromRemoteApi(2023, 10))
    print(generate_hw02("2022年台灣10月紀念日有哪些?"))
    print("====main function ends====")
