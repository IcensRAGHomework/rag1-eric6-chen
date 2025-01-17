from typing import Optional, Type

import json
import traceback
import re
import base64
from mimetypes import guess_type

from model_configurations import get_model_configuration

import requests
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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

hw04_system_prompt = """
Please reply my questions.
Format requirement:
- Replay in Traditional Chinese (zh-TW)
- Reply in JSON format.
- The JSON format is `{"Result":{"score":"integer: score you read from the given input"}}`
- Format the JSON output with indent of 4 spaces
- REPLY JSON ONLY. DO NOT REPLY WITH MARKDOWN CODE BLOCK SYNTEX.
"""


def get_llm():
    return AzureChatOpenAI(
        model=gpt_config["model_name"],
        deployment_name=gpt_config["deployment_name"],
        openai_api_key=gpt_config["api_key"],
        openai_api_version=gpt_config["api_version"],
        azure_endpoint=gpt_config["api_base"],
        temperature=gpt_config["temperature"],
    )


def generate_hw01(question):
    llm = get_llm()
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
    print(f"get_holiday_info_remote get called with arg :{year}, {month}")
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
    llm = get_llm()
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


session_store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def generate_hw03(question2, question3):
    llm = get_llm()
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

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    intput = hw02_system_prompt + "\n MY QUESTION IS: " + question2
    response1 = agent_with_chat_history.invoke(
        {"input": intput}, config={"configurable": {"session_id": "<foo>"}}
    )
    # print("response1:" + str(response1))
    q3_leading_prompt = """
    描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。
    我需要的輸出格式是：`{"Result":{"add":"boolean: 表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true，否則為 false。","reason":"string: 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。"}}`
    我的問題是：
    """
    response2 = agent_with_chat_history.invoke(
        {"input": q3_leading_prompt + question3},
        config={"configurable": {"session_id": "<foo>"}},
    )
    return str(response2["output"])


# Function to encode a local image into data URL.
# From: https://learn.microsoft.com/zh-tw/azure/ai-services/openai/how-to/gpt-with-vision?tabs=python
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def generate_hw04(question):
    image_path = r"baseball.png"
    image_data_url = local_image_to_data_url(image_path)
    # print("Data URL:", image_data_url)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hw04_system_prompt),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    }
                ],
            ),
            ("human", "{question}"),
        ]
    )
    response = llm.invoke(prompt.format(question=question))
    print(response)


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
    # print(generate_hw02("2022年台灣10月紀念日有哪些?"))
    # print(
    #     generate_hw03(
    #         """2024年台灣10月紀念日有哪些?""",
    #         """根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？""",
    #     )
    # )
    print(generate_hw04("請問中華台北的積分是多少"))
    print("====main function ends====")
