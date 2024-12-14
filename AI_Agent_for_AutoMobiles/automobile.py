import streamlit as st # type: ignore
from crewai import Agent, Task, Crew, Process # type: ignore
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool # type: ignore
from langchain.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

st.title("BikeGenie: Tailored Two-Wheeler Buying Guide with AI-V2!🏍️🤖")

question = st.text_area("Please enter your query here:",height=150)

search_tool = SerperDevTool()
#scrape_tool = ScrapeWebsiteTool()

llm = Ollama(model = 'llama3',temperature=0.5)

# Defining agents from crew ai for performing a task
adviser = Agent(
    role="Automobile Advisor",
    goal="Analyze the users query or question {question} and provide a preliminary,comprehensive and detailed advice for factors to be considered while buying bikes.",
    backstory="This agent specializes in advising automobile suggestion/advices based on the users query/question. It uses advanced algorithms and automobile knowledge to identify potential answers.",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool],
    llm=llm,
)

recommender = Agent(
    role="Automobile Proposer",
    goal="Recommend appropriate automobiles based on the query/question {question} provided.",
    backstory="This agent specializes in creating automobile suggestions tailored to individual needs. It considers the current best practices in indian automobile to recommend accurate/sensible automobiles.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
)

# Defining tasks for the agents to excute the assigned work
adviser_task = Task(
    description=(
        "1. Analyze the users's query ({question}).\n"
        "2. provide a preliminary,comprehensive and detailed advice for factors to be considered for india motorvehicles based on the provided information.\n"
        "3. Limit the advise to tailored buying guides."
    ),
    expected_output="A preliminary advises on the {question} with a list of possible buying guides.",
    agent=adviser,
)

recommender_task = Task(
    description=(
        "1. Based on the {question}, recommend appropriate indian automobile vehicle step by step.\n"
        "2. Consider the users's query/question ({question}) and undestand the current condition of the user.\n"
        "3. Provide detailed recommendations, including model summary,mileage,price,why should i choose this bike?."
    ),
    expected_output="A comprehensive automobile plan tailored to the users's {question} stasting its model summary,mileage,price and why should i choose this bike?.",
    agent=recommender,
)

# Creating crew agent 
crew = Crew(
    agents=[adviser,recommender],
    tasks=[adviser_task, recommender_task],
    process=Process.sequential
)

# Execution
submit = st.button("Get Tailored Plan")

if submit:
    with st.spinner('Generating recommendations...'):
        result = crew.kickoff(inputs={"question": question})
        st.write(result)