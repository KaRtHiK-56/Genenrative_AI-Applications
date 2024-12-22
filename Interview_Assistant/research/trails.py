import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA



loader = PyPDFLoader(r"C:\Users\Devadarsan\Desktop\Karthik_projects\Interview_Assistant\data\ml.pdf")
loader = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=300)
splitter = splitter.split_documents(loader)

embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

)
db = FAISS.from_documents(splitter, embeddings)


llm = Ollama(model='llama3',temperature=0.5)

prompt_template = """
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the coders or programmers for their tests.
Make sure not to lose any important information.

QUESTIONS:
"""
prompt = PromptTemplate(template=prompt_template,input_variables=['text'])

# this refined prompt is to extract best of the LLM output 
refined_template = ("""
You are an expert at creating practice questions based on coding material and documentation.
Your goal is to help a coder or programmer prepare for a coding test.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)

refined_template = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refined_template,
)

from langchain.chains import load_summarize_chain
ques_gen_chain = load_summarize_chain(llm = llm, 
                                          chain_type = "refine", 
                                          verbose = True, 
                                          question_prompt=prompt, 
                                          refine_prompt=refined_template)

question = ques_gen_chain.run(splitter)

print(question)

vector_store = FAISS.from_documents(splitter, embeddings)
llm_answer_gen = Ollama(temperature=0.1, model="llama3")

question
question_list = question.split('\n')



answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                               chain_type="stuff", 
                                               retriever=vector_store.as_retriever())

# Answer each question and save to a file
for question in question_list:
    print("Question: ", question)
    answer = answer_generation_chain.run(question)
    print("Answer: ", answer)
    print("--------------------------------------------------\\n\\n")
    # Save answer to file
    with open("answers.txt", "a") as f:
        f.write("Question: " + question + "\\n")
        f.write("Answer: " + answer + "\\n")
        f.write("--------------------------------------------------\\n\\n")