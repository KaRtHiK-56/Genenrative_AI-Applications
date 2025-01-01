import streamlit as st
from langchain.prompts import PromptTemplate 
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma 
from PyPDF2 import PdfReader
from datetime import datetime
from langchain.chains.summarize import load_summarize_chain



st.title("Research paper Blog Generator")
doc = st.file_uploader("Please upload your document here:",type='pdf')

def reblog(doc):
    docs = PdfReader(doc)
    print("*"*60)
    print("Document loaded successfully!")

    text = ""
    for page in docs.pages:
        text += page.extract_text()
    print('text is ',text)

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
    splitter = splitter.split_text(text)
    print("*"*60)
    print("splitting of the document is successful!")

    #embeddings = OllamaEmbeddings(model='nomic-embed-text')

    #start = datetime.now()
    #print('start time is:',start)
    #db = Chroma.from_texts(splitter,embeddings,persist_directory='reblog_db')
    #db.persist()
    #end = datetime.now()
    #print('end time is:',end)
    #print("*"*60)
    #print("embedding + storing is completed!")
#
    #reteriver = db.as_retriever()
#
    prompter = """   
                You are a highly skilled writer specializing in transforming complex research papers into clear, engaging, 
                and comprehensive articles. Your task is to break down the provided research paper {splitter} into an article that covers
                every key concept, method, result, and conclusion presented in the {splitter}. The article should be structured 
                in a blog format, making it easy for readers to understand and absorb the information from the document,dont make up things.

                {splitter}

                Please follow these guidelines:

                1. Title: Create a catchy and informative title for the article.
                2. Introduction: Write a brief introduction that provides an overview of the research topic, its significance, and the primary objectives of the study.
                3. Main Topics:
                    - Identify and outline the major sections of the research paper.
                    - For each section, create a subheading and provide a detailed explanation of the content.
                    - Include any important figures, tables, or data from the research paper, and explain their relevance.
                4. Methods: Describe the methodologies used in the research in a simplified manner, ensuring that readers understand the process and its importance.
                5. Results: Summarize the key findings of the research, highlighting significant data points and outcomes.
                6. Discussion: Discuss the implications of the results, how they contribute to the field, and any potential limitations or areas for future research.
                7. Conclusion: Write a concise conclusion that summarizes the main points of the article, reinforcing the importance of the research and its contributions.

                Make sure to use clear, concise language and avoid jargon as much as possible. The goal is to make the research paper accessible to a broader audience while maintaining the integrity and accuracy of the original content.
                
"""

    prompt = PromptTemplate(template=prompter)
    prompt = prompt.format(splitter=splitter)
    
    llm = Ollama(model='llama3',temperature = 0.05)

    response = llm(prompt)
    print(response)

    return response

    
    

submit = st.button("Generate")

if submit:
    with st.spinner("Generating.........."):
        st.write(reblog(doc))