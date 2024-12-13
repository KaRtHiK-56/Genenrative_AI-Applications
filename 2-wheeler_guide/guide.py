import streamlit as st 
from langchain_community.llms import Ollama 
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import LLMChain
from langchain.chains import create_retrieval_chain
import bs4
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.chains.router import MultiPromptChain

#st.title("2-Wheeler Guide 🏍️")
#question = st.text_area("Please enter your query:",height=150)

#loading the necessary documents 
text = TextLoader(r'C:\Users\Devadarsan\Desktop\Karthik_projects\2-wheeler_guide\guide.txt')
text = text.load()

web1 = WebBaseLoader(web_paths=("https://auto.hindustantimes.com/new-bikes/bikes-under-50000",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("finderCard")
)))
web2 = WebBaseLoader(web_paths=("https://auto.hindustantimes.com/new-bikes/search?budget=50000-k-1-lakhs",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("finderCard")
)))
web3 = WebBaseLoader(web_paths=("https://www.bikewale.com/new-bike-search/best-bikes-under-1-lakh/?sortField1=modelPopularity&sortOrder1=desc&budget=100000%2C150000",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("o-fznJzb o-fHmpzP")
)))
web4 = WebBaseLoader(web_paths=("https://auto.hindustantimes.com/new-bikes/search?budget=1-lakhs-2-lakhs",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("finderCard")
)))
web5 = WebBaseLoader(web_paths=("https://auto.hindustantimes.com/new-bikes/search?budget=2-3-lakhs",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("finderCard")
)))
web6 = WebBaseLoader(web_paths=("https://auto.hindustantimes.com/new-bikes/search?budget=3-4-lakhs",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("finderCard")
)))
web7 = WebBaseLoader(web_paths=("https://auto.hindustantimes.com/new-bikes/search?budget=4-5-lakhs",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("finderCard")
)))
web1=web1.load()
web2=web2.load()
web3=web3.load()
web4=web4.load()
web5=web5.load()
web6=web6.load()
web7=web7.load()


# splitting the documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splitter = splitter.split_documents(text)
#embedding the data and storing that in the vectorstore 
embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True})

db = Chroma.from_documents(splitter, embeddings)

#defining the llama3 model 
llm = Ollama(model="llama3",temperature=0)

#creating LLM router template
advance_prompt = """ You are an expert 2 wheeler bike reviewer/guide/resource person.
                                              you know in and out everything about 2 wheeler motor cycle.
                                           Provide comprehensive, sensible, detailed, and well-adapted answers to user queries. 
                                          Ensure high accuracy and relevance to the specific question and situation.
                                          <context>   
                                          {context}
                                          </context>
                                          question:{input}
                                          
                                          I need the output to be in the format of
                                          ["Model": "", /n/n
                                            "Summary": "" /n/n
                                            "Mileage ": "",/n/n
                                            "Price": "", /n/n
                                            "why should i buy this bike " : ""]   """

beginner_prompt = """
You are an expert 2 wheeler bike reviewer/guide, you know in and out of everything in indian Bikes.
your role is to help people with their query regarding guide/factors to purchase the bikes.
Here is the question <context>   
                    {context}
                    </context> 
                    question:{input}
"""

# creating a router prompt template 
prompt_infos = [

    { "name":"advance" ,
     "description":"Answering question related to suggesting bikes" ,
     "template":advance_prompt, },

     { "name":"beginner" ,
     "description":"Answering questions for factors/guides for purchasing bikes" ,
     "template": beginner_prompt, },
]

destination_p = {}
for info in prompt_infos:
    name = info['name']
    prompt_template = info['template']
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm = llm,prompt = prompt)
    destination_p['name'] = chain

default_prompt = ChatPromptTemplate.from_template('{input}')
default_chain = LLMChain(llm=llm,prompt=default_prompt)

destination = [f"{p['name']}:{p['description']}" for p in prompt_infos] 
destination_str = "/n".join(destination)

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destination_str)

router_prompt = PromptTemplate(template=router_template,input_variables=['input'],output_parser=RouterOutputParser())

router_chain = LLMRouterChain.from_llm(llm,router_prompt)

chain = MultiPromptChain(router_chain=router_chain,destination_chains=destination_p,default_chain=default_chain,verbose=True,silent_errors=True)

retriver = db.as_retriever()

chains = create_retrieval_chain(retriver,chain)
answer = chains.invoke({"input":"what should i buy 400cc or 600cc?"})
print("answer is",answer['answer'])

