import datetime
from dotenv import load_dotenv

load_dotenv()
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from reflexionSchema import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI()
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actorPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
        Current time: {time}
        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. Recommend search queries to research information and improve your answer.
        """,
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's "),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actorPromptTemplate.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """ Revise your previous answer using the new information.
                        - You should use the previous critique to add important information to your answer.
                        - You must include numerical citations in your revised answer to ensure it can be verified.
                        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In the form of 
                            - [1] https://example.com
                            - [2] https://example.com
                        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words
"""

revision_prompt_template = actorPromptTemplate.partial(
    first_instruction=revise_instructions
)

revisor = revision_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC/ autonomous soc problem domain,"
        "list startups that do that and raised capital"
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
