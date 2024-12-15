import streamlit as st 
from langchain_community.llms import Ollama 
from crewai import Agent,Task,Crew,Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')
print("Serper api key loaded successfully!!!")

st.set_page_config(page_title='INV AGENT',page_icon='🤖')
st.title("AI AGENTS FOR PROPERTY INVESTEMENTS🤖")

print("Inside function")
search_tool = SerperDevTool()
llm = Ollama(model='llama3',temperature=0)
researcher = Agent(
    role="Expert Property Researcher",
    goal="Find promising investment properties.",
    backstory="You are a veteran property analyst. In this case you're looking for retail properties to invest in.",
    llm =llm,
    allow_delegation=False,
    tools=[search_tool],
    verbose=True,
)
print("researcher defined successfully!!!")
task_1 = Task(
    description="Search the internet and find 5 promising real estate investment suburbs in chennai . For each suburb highlighting the mean, low and max prices as well as the rental yield and any potential factors that would be useful to know for that area.",
    expected_output="""A detailed report of each of the suburbs.The results should be formatted as shown below: 

    Suburb 1: Chennai
    Price of the property: Rs:1,12,00,000
    Rental Vacancy: 4.2%
    Rental Yield: 2.9%
    Background Information: These suburbs are typically located near major transport hubs, employment centers, and educational institutions. The following list highlights some of the top contenders for investment opportunities """,
    agent=researcher,
)
print("task 1 defined successfully!!!")

writer = Agent(
    llm=llm,
    role="Senior Property Analyst",
    goal="Summarise property facts into a report for investors.",
    backstory="You are a real estate agent, your goal is to compile property analytics into a report for potential investors.",
    allow_delegation=False,
    verbose=True,
)
print("writer defined successfully!!!")
task_2 = Task(
    description="Summarise the property information into bullet point list. ",
    expected_output="A summarised dot point list of each of the suburbs, prices and important features of that suburb.",
    agent=writer,
)
print("task defined successfully!!!")
crews = Crew(
    agents=[researcher,writer],
    tasks = [task_1,task_2],
    verbose = 2,
    process=Process.sequential
)

print("Crew defined successfully!!!")
result = crews.kickoff()

st.write(result)


