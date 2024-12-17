import streamlit as st  # type: ignore
from crewai import Agent,Task,Process,Crew # type: ignore
from crewai_tools import SerperDevTool
from langchain.llms import Ollama
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

search_tool = SerperDevTool()

llm = Ollama(model='llama3',temperature=0.6)

st.title("Your Technical Blogger Agent")
topic = st.text_area("Tell me on what topic I should write a Blog post on....",height = 100)


def blog(topic):
    researcher = Agent( 
    role = " Senior expertise researcher ",
    goal = " To research the context/contents on the {topic}.  ",
    backstory = ("""
    As a senior research expertise dedicated to uncovering the most impactful trends,
    you're propelled by a relentless curiosity and a commitment to innovation. 
    Your role involves delving deep into the latest developments 
    across various sectors to identify and analyze 
    the top trending news within any given field/domain/technology. 
    This pursuit not only satisfies your thirst for knowledge
    but also enables you to contribute valuable insights that could 
    potentially reshape understandings and expectations on a global scale.
    """),
    verbose = True,
    allow_deligation = True,
    tools = [search_tool],
    llm = llm,)

    r_task = Task(
        description = """ 
        Expertise in creating a detailed and comprehensive content/context on the {topic}
        Focus on identifying pros and cons and the overall narrative.
        Your final report should clearly articulate the key points
        """,
        expected_output = " To provide the context in a detailed and comprehensive 1000 words contents on the {topic} ",
        tools = [search_tool],
        agent = researcher,
    )


    writer = Agent(
        role = " Senior/Creative Expertise Writer/Blogger",
        goal = " To write compelling and technical contents/articles about {topic} ",
        backstory = (''' 
        Armed with the knack for distilling complex subjects into digestible,
        compelling stories, you, as a blog writer, masterfully weave narratives 
        that both enlighten and engage your audience. Your writing illuminates fresh 
        insights and discoveries, making them approachable for everyone. Through your craft,
        you bring to the forefront the essence of new developments across various topics, 
        making the intricate world of news a fascinating journey for your readers.
    '''),
        verbose = True,
        allow_deligation = False,
        tools = [search_tool],
        llm = llm,
    )

    w_task = Task(
        description = (""" Compose an insightful,detaiuled and technical article on {topic}."
        Focus on the technical perspective,ho to guides and coding examples .
        This article should be easy to understand, engaging, and positive. """),
        expected_output = (""" A detailed and comprehensive article covering the topics
    Introduction,
    Architecture,
    Components,
    Future Enhancements,
    Benefits,
    Drawbacks,
    Conclusion.
    """),
        tools = [search_tool],
        agent = researcher,
        aync_execution=False,
        output_file="blog_post.md"
    )

    crew = Crew(
        agents = [researcher,writer],
        tasks = [r_task,w_task],
        verbose = 2,
        process = Process.sequential,
    )
    response = crew.kickoff(inputs={"topic":topic})

    return response

submit = st.button("Create")
if submit:
    with st.spinner("Creating the content"):
        st.write(blog(topic))