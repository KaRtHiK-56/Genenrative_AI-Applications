import streamlit as st
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

st.title("TEXT-2-SQL CONVERTOR TOOL 🔁")
text = st.text_area("Please enter/paste your sql query in natural language to convert that into SQL Query:",height=100)



def convertor(sql):
    prompts = """ create a sql query using the below snippet
    '''{sql}'''
    Just SQL query:
    """
    prompt_template = PromptTemplate.from_template(template=prompts)
    llm = Ollama(model='llama3',temperature = 0)
    chain = LLMChain(prompt=prompt_template,llm=llm)
    response = chain.invoke(sql)
    return response


convert = st.button("Convert")
if convert:
    st.write(convertor(text))


