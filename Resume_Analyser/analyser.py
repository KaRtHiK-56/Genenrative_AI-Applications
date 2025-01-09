import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama


st.title("RESUME ANALYSER")
st.header("This resume analyser application will compare your resume against the job description provided")
jd = st.text_area("Please Enter/paste your job description here:",height=250)
resume = st.file_uploader("Please upload your resume/CV here:")


prompts = """
    Hey Act Like a skilled or very experience ATS(Application Tracking System)
    with a deep understanding of tech field, python, python developer, software engineering, data science,
    data analyst,machine learning, and big data engineer. Your Task is to evaluate the resume based on the
    given job description.
    You must consider the job market is very competitive and you should provide
    best assistance for the improving the resume. Assisn the percentage Matching
    based on JD(Job Description),skill match out of 10, and the missing keywords with high accuracy 
    resume:{resume}
    description:{jd}

    I want the response in the structure
    ["JD Match":  "%", /n/n
    "Skill match": " out of 10" /n/n
    "MissingKeywords : []",/n/n
    "Profile Summary" : ""]

"""

def analyser(resume,jd):
    prompt_template = PromptTemplate.from_template(template=prompts)
    prompt = prompt_template.format(resume=resume,jd=jd)
    llm = Ollama(model="llama3",temperature=0)
    response = llm(prompt)
    return response

analyse = st.button("Analyse")

if analyse:
    st.write(analyser(resume,jd))