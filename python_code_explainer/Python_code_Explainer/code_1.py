from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
import streamlit as st 

#creating the UI of the code 
st.title("Python Code Explainer")
code = st.text_area("Please type/paste your python code:",height=250)

#few shots prompts technique
input = """
Code snippet 1: 
def fibonacciSeries(i):
	    if i < = 1:
		    return i
	    else:
		    return (fibonacciSeries(i - 1) + fibonacciSeries(i - 2))
    num=10
    for i in range(num):
	    print(fibonacciSeries(i), end=" ")

Output : if i = 5 
0 1 1 2 3

Algorithm :
Add integer value to num.
Then run a for loop where i is from range 0 to num.
In each iteration print and call the fibonacciSeries function with i as a parameter.
In the Recursive function fibonacciSeries,
Check if i <= 1, if it is True then return i
Else return fibonacci(i - 1) + fibonacci(i - 2).


Code snippet 2: 
def modify_list(input_list):
  input_list.append(4)
  input_list = [1,2,3]
my_list = [0]
modify_list(my_list)
print(my_list)

Output : [0, 4]

Algorithm :
Inside the modify_list function, an element 4 is appended to input_list.
Then, input_list is reassigned to a new list [1,2,3], but this change doesn't affeact the original list.
So, print(my_list) outputs [0, 4].
"""
prompter = """
  Your are a well experienced senior python developer.
  Your Task is to act as Python Code Explainer.
  I'll give you a Code Snippet.
  Your Job is to explain the Code Snippet step-by-step in detailed and elobrated way.
  Break down the code into as many steps as possible and also understand what inputs and outputs it generates.
  Share intermediate checkpoints & steps along with results.
  If there is no input provided , create your on input according to the context of the code.
  Few good examples of python code output between #### seperator:
  ####
  {input}
  ####
  Code Snippet is shared below, delimited with triple backticks:
  ```
  {code}
  ```
"""
#creating the python function for generating the email response using LLM
def code_explainer(codes,inputs):
    prompt_template = PromptTemplate.from_template(template=prompter)
    prompt = prompt_template.format(code=codes,input=inputs)
    llm = Ollama(model="llama3",temperature=0)
    response = llm(prompt)
    return response


#ceating the submit button and passing the data after submit button is clicked
submit = st.button("Generate")

if submit:
    st.write(code_explainer(code,input))