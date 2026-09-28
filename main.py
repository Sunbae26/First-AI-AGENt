import os
import certifi
import requests


from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
W_STACK = os.getenv("W_STACK")

search_tool = TavilySearchResults(max_results=3)


@tool
def get_weather(city: str) -> str:
    """Fetch current weather for a city."""
    response = requests.get(
        "http://api.weatherstack.com/current",
        params={"access_key": W_STACK, "query": city},
        timeout=15,
    )
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather for {city}: {data.get('error', {}).get('info', 'unknown error')}"

    return f"City: {city}\nTemperature: {data['current']['temperature']} Celsius"


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the available tools whenever you need current information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [search_tool, get_weather]

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke({
    "input": "Find the capital of India and its current weather"
})

print(response["output"]) 