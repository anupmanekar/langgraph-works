import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_fireworks import ChatFireworks

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph

class OverallState(TypedDict):
    subject: str
    topics: list
    questions: Annotated[list, operator.add]
    top_5_questions: list[str]

class Topics(BaseModel):
    topics: list[str]

class Questions(BaseModel):
    questions: list[str]

class OutputQuizState(BaseModel):
    questions: List[str]

llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x22b-instruct") 

# Define two prompts for topics_generation and questions_generation

topics_generation = "You are experienced web searcher. Please find 5 interesting topics related to the subject: {subject}."
questions_generation = "You are experienced web searcher. Please find 5 interesting questions related to the topic: {topic}."

# Define two nodes generate_topics and generate_questions_by_topics
# generate_topics: takes a subject and returns a list of topics
# generate_questions_by_topics: takes a topic and returns a list of questions

def generate_topics(state: OverallState):
    subject = state['subject']
    response = llm.with_structured_output(Topics, include_raw=False).invoke(topics_generation.format(subject=subject))
    return {"topics": response.topics}

def multiple_questions_generation(state: OverallState):
    return [Send("generate_questions_by_topics", {"topic": t}) for t in state["topics"]]

def generate_questions_by_topics(topic: str):
    response = llm.with_structured_output(Questions, include_raw=False).invoke(questions_generation.format(topic=topic))
    return {"questions": response.questions}

def reshuffle_questions(state: OverallState):
    # Shuffle the questions
    sorted_questions = sorted(state["questions"], key=lambda x: hash(x))
    # Get top 5 questions
    return {"top_5_questions": sorted_questions[:5]}

# Define the state graph
quiz_generator_graph = StateGraph(state_schema=OverallState)

quiz_generator_graph.add_node("generate_topics", generate_topics)
quiz_generator_graph.add_node("reshuffle_questions", reshuffle_questions)
quiz_generator_graph.add_node("generate_questions_by_topics", generate_questions_by_topics)
quiz_generator_graph.add_edge(START, "generate_topics")
quiz_generator_graph.add_conditional_edges("generate_topics", multiple_questions_generation, ["generate_questions_by_topics"])
quiz_generator_graph.add_edge("generate_questions_by_topics", "reshuffle_questions")
quiz_generator_graph.add_edge("reshuffle_questions", END)

graph_instance = quiz_generator_graph.compile()