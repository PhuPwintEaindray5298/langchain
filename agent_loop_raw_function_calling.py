from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

MAX_INTERATION = 10
MODEL = "qwen3.6:27b"

# Tools (Langchain @tool decorator)

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"Executing get_product_price(product='{product}')")
    prices = {"laptop":1299.99, "headphones":149.95, "keyboard":89.5}
    return prices.get(product,0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price. Available tiers: bronze, silver and gold"""
    print(f"Executing apply_discount(price='{price}',discount_tier='{discount_tier}')")
    discount_percentages = {"bronze":5, "silver":12, "gold":23}
    discount = discount_percentages.get(discount_tier,0)
    return round(price*(1-discount/100),2)

tools_for_llm = [
    {
        "type":"function",
        "function":{
            "name":"get_product_price",
            "description":"Look up the price of a product in the catalog",
            "parameters":{
                "type":"object",
                "properties":{
                    "product":{
                        "type":"string",
                        "description":"The product name, e.g. 'laptop','headphones','keyboard'",
                    },
                },
                "required":["product"],
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"apply_discount",
            "description":"Aplly a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters":{
                "type":"object",
                "properties":{
                    "price":{"type":"number","description":"The original price"},
                    "discount_tier":{
                        "type":"string",
                        "description":"The discount tier: 'bronze','silver', or 'gold'"
                    },
                },
                "required":["price","discount_tier"],
            },
        },
    }
    
]

@traceable(name="Ollama Chat",run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL,tools=tools_for_llm,messages=messages)

# Agent Loop
@traceable(name="Ollama Agent Loop")
def run_agent(question:str):
    tools_dict = {
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }
 
    print(f"Question: {question}")
    print("="*40)
    messages = [
        {   "role":"system",
            "content":
             "You are a helpful shopping assistant. "
             "You have access to a product catalog tool and a discount tool.\n\n"
             "Strict Rules - you must follow these exactly:\n"
             "1. NEVER guess or assume any product price. You must call get_product_price first to get the real price.\n"
             "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price"
             "returned by get_product_price - do NOT pass a made-up number.\n"
             "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.\n"
             "4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one."
            
         },
        {"role":"user","content":question}
    ]
    
    for iteration in range(1,MAX_INTERATION+1):
        print(f"\n Iteration {iteration}---")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        #If no tool calls, this is an answer
        if not tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content
    
    #Process only the first tool call -force one tool per iteration
    tool_call = tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = tool_call.function.arguments
   
    print(f"[Tool Selected] {tool_name} with args: {tool_args}")

    tool_to_use = tools_dict.get(tool_name)

    if tool_to_use is None:
        return ValueError(f"Tool '{tool_name}' not found")
    
    observation = tool_to_use(**tool_args)
    print(f"[Tool Result] {observation}")

    messages.append(ai_message)
    messages.append({"role":"tool","content":str(observation)})

    
    return None


if __name__=="__main__":
    print("Hello LangChain AGent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
