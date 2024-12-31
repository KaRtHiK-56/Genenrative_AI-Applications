# creates an AI-powered virtual vet assistant
# called "AI Pet-Vet". Let's break down the imports and their functionalities:
import streamlit as st
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool


# setting up the page configuration for the AI Pet-Vet application
# using Streamlit, a popular Python library for creating web applications with simple Python scripts.
st.set_page_config(page_title="Pet-Vet", page_icon="🐶")

st.title("🤖 AI Pet-Vet 🐶")
st.text(
    "Ask questions related to your pet injuries and behaviours,this virtual vet doc will help you clearing your basic needs."
)
question = st.text_area("Please enter your question here:")


def vet(question):
    """
    The `vet` function utilizes agents and tasks to search through a veterinary medicine PDF, extract
    relevant information, and generate professional emails based on the findings.
    
    Args:
      question: The `vet` function you provided seems to be a script for a virtual assistant system that
    involves searching through a veterinary medicine textbook PDF to find answers to customer questions
    and then writing professional emails based on the findings.
    
    Returns:
      The `vet` function defines a process involving a research agent and a professional writer to
    handle tasks related to answering customer questions based on a PDF document and writing
    professional emails to vet doctors. The function takes a question as input, initiates the agents and
    tasks, and then kicks off the process with the provided question. The result of this process, which
    includes answering the customer's question and writing a professional
    """


    doc = PDFSearchTool((
        #r"C:\Users\Devadarsan\Desktop\Karthik_projects\RAG_with_Agents\veterinary-medicine-a-textbook-of-the-diseases-of-cattle-horses-sheep-pigs-and-goats11th-edition.pdf"
    ),
    config=dict(
        llm=dict(
            provider="ollama", 
            config=dict(
                model="llama3",
                temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="ollama", 
            config=dict(
                model="nomic-embed-text",
                # title="Embeddings",
            ),
        ),
    )
)


    researcher = Agent(
        role="Research Agent",
        goal="Search through the PDF to find relevant answers",
        allow_delegation=False,
        verbose=True,
        backstory=(
            """
            The research agent is adept at searching and 
            extracting data from documents, ensuring accurate and prompt responses.
            """
        ),
        tools=[doc],
    )

    write = Agent(
        role="Professional Writer",
        goal="Write professional emails based on the research agent's findings",
        allow_delegation=False,
        verbose=True,
        backstory=(
            """
            The professional writer agent has excellent writing skills and is able to craft 
            clear and concise emails based on the provided information.
            """
        ),
    )

    answer_customer_question_task = Task(
        description=(
            """
        Answer the customer's questions based on the PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the  PDF.

        Here is the customer's question:
        {question}
        """
        ),
        expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the PDF.
        """,
        tools=[doc],
        agent=researcher,
    )

    write_email_task = Task(
        description=(
            """
            - Write a professional email to a vet doctor based 
                on the research agent's findings.
            - The email should clearly state the issues found in the specified section 
                of the report and request a quote or action plan for fixing these issues.
            - Ensure the email is signed with the following details:

                Best regards,

            """
        ),
        expected_output="""
                Write a clear and concise email that can be sent to a doctor and the 
                issues found in the report.
                """,
        agent=write,
    )

    # The line `crew = Crew(agents=[researcher, write], tasks=[answer_customer_question_task,
    # write_email_task], process=Process.sequential)` is creating a crew of agents and tasks to handle
    # the process of answering customer questions based on a PDF document and writing professional
    # emails to vet doctors.
    crew = Crew(
        agents=[researcher, write],
        tasks=[answer_customer_question_task, write_email_task],
        process=Process.sequential,
    )

    result = crew.kickoff(inputs={"customer_question": question})
    return result


submit = st.button("Ask")

if submit:
    with st.spinner("Thinking...."):
        st.write(vet(question))
