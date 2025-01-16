import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

gpt_chat_version = "gpt-4o"
gpt_config = get_model_configuration(gpt_chat_version)
llm = AzureChatOpenAI(
    model=gpt_config["model_name"],
    deployment_name=gpt_config["deployment_name"],
    openai_api_key=gpt_config["api_key"],
    openai_api_version=gpt_config["api_version"],
    azure_endpoint=gpt_config["api_base"],
    temperature=gpt_config["temperature"],
)

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


def generate_hw01(question):
    prompt = ChatPromptTemplate.from_messages(
        [("system", hw01_system_prompt), ("human", "{question}")]
    )
    return llm.invoke(prompt.format(question=question, format=hw01_json_format))


def generate_hw02(question):
    pass


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
    print("main function start")
    pass
    print("main function ends")
