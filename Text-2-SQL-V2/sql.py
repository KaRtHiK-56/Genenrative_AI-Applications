import streamlit as st
import sqlite3
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate


st.title("TEXT-2-SQL using DB 🔁")
question = st.text_area("Please enter/paste your sql query in natural language to convert that into SQL Query:",height=100)

def sql_connection(sql,db):
    print("inside try")
    conn = sqlite3.connect(db)
    c = conn.cursor()
    print("c",c)
    c.execute(sql)
    results = c.fetchall()
    print("results",results)
    conn.commit()
    conn.close()
    return results
        


prompts = """
You are an expert in SQL and your task is to convert natural language queries into precise SQL code. Given the following natural language query, generate the corresponding SQL statement. The output should only include the SQL code without any additional text.
these are the table column name available in the employee table which has these columns emp_id, name, age, gender, salary.
Natural Language Query: "{question}"

SQL Code:

"""
    
def convertor(question,prompt):
    prompt_template = PromptTemplate.from_template(template=prompt)
    prompt = prompt_template.format(question=question)
    llm = Ollama(model='llama3',temperature = 0.5)
    response = llm(prompt)
    return response


convert = st.button("Convert")
if convert:
    response1 = convertor(question,prompts)
    print("reponse",response1)
    response2 = sql_connection(response1,r"C:\Users\Devadarsan\Desktop\Karthik_projects\Text-2-SQL-V2\data.db")
    for row in response2:
        print(row)
        st.header(row)

