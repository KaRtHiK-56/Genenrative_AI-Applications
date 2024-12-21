import streamlit as st
from langchain_community.llms import Ollama 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent


st.set_page_config(page_title='analyst',page_icon="📈")
st.header("🤖 Ask AI Agent Data Analyst 📈")
file = st.file_uploader("Please upload your csv data here:",type="csv")

def analyst():
    llm = Ollama(model='llama3',temperature=0.7)
    if file is not None:
        agent = create_csv_agent(
                llm,
                file,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                handle_parsing_errors=True,
        )

        question = st.text_area("Please ask your question:")

        submit = st.button("Ask!")
        if submit:
             if question is not None and question != "":
                with st.spinner(text="In progress..."):
                    st.write(agent.run(question))

if __name__ == "__main__":
    analyst()




