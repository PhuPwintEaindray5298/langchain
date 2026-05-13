from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable

MAX_INTERATION = 10
MODEL = "qwen3.6:27b"

# Tools (Langchain @tool decorator)

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"Executing get_product_price(product='{product}')")
    prices = {"laptop":1299.99, "headphones":149.95, "keyboard":89.5}
    return prices.get(product,0)

@tool
def apply_discount(price: float, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price. Available tiers: bronze, silver and gold"""
    print(f"Executing apply_discount(price='{price}',discount_tier='{discount_tier}')")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price*(1-discount/100),2)

# Agent Loop
@traceable(name="LangChain Agent Loop")
def run_agent(question:str):
    tools = [get_product_price,apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}",temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("="*40)
    messages = [
        SystemMessage(content="You are a helpful shopping assistant.You have access to a product catalog tool and discount tool Strict rules - you must follow these exactly:\n" \
        "Only call apply_discount after you have recieved a price from get_product_price. Pass the exact price. Never calculate discount yourself using math. Alway use the apply_discount tool.\n" \
        "If the user does not specify the discount tier, ask them which tier to use - do Not assume one"),
        HumanMessage(content=question)
    ]
    
    for iteration in range(1,MAX_INTERATION+1):
        print(f"\n Iteration {iteration}---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        #If no tool calls, this is an answer
        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content
    
        #Process only the first tool call -force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")
        print(f"[Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            return ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"[Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("Error: Max iterations reached without a final answer")
    return None

if __name__=="__main__":
    print("Hello LangChain AGent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
