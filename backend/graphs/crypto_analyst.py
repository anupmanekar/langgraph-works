# Agentic script to generate best buying and selling prices for a cryptocurrency from a given list of exchanges using Tavily API
import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_fireworks import ChatFireworks

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph


def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return [*left, *right]

class OverallState(TypedDict):
    message: str
    token: str
    exchanges: list[str]
    buying_prices: Annotated[list, reduce_list]
    selling_prices: Annotated[list, reduce_list]
    best_selling_price: float
    best_buying_price: float

class TokenExchanges(BaseModel):
    token: str
    exchanges: list[str]

class PriceStruct(BaseModel):
    exchange: str
    buying_price: float
    selling_price: float

class TokenExchangeMapping(BaseModel):
    token: str
    exchange: str

llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x22b-instruct")

TOKEN_EXCHANGES_INFERENCE = """ You are an experienced crypto analyst and are tasked with fetching real time prices for token from particular exchange.
    Please infer the requested token and exchanges from the given message.
    Prices should only be in USDT and NOT in BTC or ETH.
    Please note that we are only considering following exchanges: Binance, Coinbase, Kraken, Bitfinex, and Uniswap.
    if you are unable to infer the token and exchanges, please ask for more information.
    """

FETCH_RATES = "You are an experienced crypto analyst. Please fetch the rates for the token: {token} on the exchange: {exchange}."

def infer_token_and_exchanges(state: OverallState):
    system_message = SystemMessage(TOKEN_EXCHANGES_INFERENCE)
    human_message = HumanMessage(state["message"])
    response = llm.with_structured_output(TokenExchanges, include_raw=False).invoke([system_message, human_message])
    return {"token": response.token, "exchanges": response.exchanges}

def fetch_rates_from_multiple_exchanges(state: OverallState):
    return [Send("fetch_rate_from_exchange", {"token": state["token"], "exchange": ex}) for ex in state["exchanges"]]

def fetch_rate_from_exchange(token_exchange: TokenExchangeMapping):
    # search = TavilySearchAPIWrapper()
    # tavily_tool = TavilySearchResults(api_wrapper=search)
    # llm_with_tools = llm.bind_tools(tavily_tool)
    response = llm.with_structured_output(PriceStruct, include_raw=False).invoke(FETCH_RATES.format(token=token_exchange["token"], exchange=token_exchange["exchange"]))
    return {"buying_prices": [response.buying_price], "selling_prices": [response.selling_price]}

def find_best_prices(state: OverallState):
    # Find the best buying and selling prices
    best_selling_price = max(state["selling_prices"])
    best_buying_price = min(state["buying_prices"])
    return {"best_selling_price": best_selling_price, "best_buying_price": best_buying_price}

# Define the state graph
crypto_analyst_graph = StateGraph(state_schema=OverallState)

crypto_analyst_graph.add_node("infer_token_and_exchanges", infer_token_and_exchanges)
crypto_analyst_graph.add_node("fetch_rate_from_exchange", fetch_rate_from_exchange)
crypto_analyst_graph.add_node("find_best_prices", find_best_prices)
crypto_analyst_graph.add_edge(START, "infer_token_and_exchanges")
crypto_analyst_graph.add_conditional_edges("infer_token_and_exchanges", fetch_rates_from_multiple_exchanges, ["fetch_rate_from_exchange"])
crypto_analyst_graph.add_edge("fetch_rate_from_exchange", "find_best_prices")
crypto_analyst_graph.add_edge("find_best_prices", END)

graph_instance = crypto_analyst_graph.compile()