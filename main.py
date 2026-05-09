from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

class Source(BaseModel):
    """ Schema for a source used by the agent
    """
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer for the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

tavily = TavilyClient()
@tool
def search(query: str) -> str:
    """
    Tool that searches over internet.
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

def langchain_introduce(llm: ChatOpenAI) -> str:
    information = """
        Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. 
        Musk has been the wealthiest person in the world since 2025; as of May 2026, Forbes estimates his net worth to be US$788 billion.

        Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there.
        He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. 
        In 1995, Musk co-founded the software company Zip2. 
        Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. 
        Musk also became an American citizen in 2002.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"],template=summary_template)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})
    print(response.content)

def main():
    print("Hello from langchain-course!")
    
    llm = ChatOpenAI(temperature=0, model="gpt-5")
    tools = [search]
    agent = create_agent(model=llm,tools=tools,response_format=AgentResponse)
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Tokyo?")})
    print(result)

if __name__ == "__main__":
    main()
