import streamlit as st
from langchain.llms import Ollama 
from langchain.prompts import PromptTemplate

st.title("Python Code Generator 🐍")
question = st.text_area("Please tell the statement to geneate the python code:")

example = """   
Question 1:
write me a python code for fibonocci series

Code snippet 1:
def fibonacci(n):
    '''
    Generate a Fibonacci series up to the nth number.

    Parameters:
    n (int): The number of Fibonacci numbers to generate.

    Returns:
    list: A list containing the Fibonacci series up to the nth number.
    '''
    series = []
    a, b = 0, 1
    while len(series) < n:
        series.append(a)
        a, b = b, a + b
    return series

n = 10  # Number of Fibonacci numbers to generate
fib_series = fibonacci(n)
print(f"Fibonacci series up to {n} numbers: {fib_series}")

Question 2:
Build a pyramid using python 

code snippet 2:
floors = 3
h = 2*floors-1
for i in range(1, 2*floors, 2):
    print('{:^{}}'.format('*'*i, h))

Question 3:
Program to check a number is Armstrong or not in python programming language:

Code snippet 3:
i = 0
result = 0
n = int(input("Please enter a number: "))
number1 = n
temp = n
while n != 0:
    n = n // 10
    i += 1
while number1 != 0:
    n = number1 % 10
    result += pow(n, i)
    number1 = number1 // 10
if temp == result:
    print("The number is an Armstrong number")
else:
    print("The number is not an Armstrong number")
"""

prompter = '''
  Your are a well experienced senior python developer.
  Your Task is to act as Python Code Generator.
  I'll give you a python question.
  Your Job is to generate the Code Snippet step-by-step in detailed and elobrated way.
  Break down the code into as many steps as possible and also understand what inputs and outputs it generates.
  Share intermediate checkpoints & steps along with results.
  If there is no input provided , create your on input according to the context of the code.
  If any question apart from python is asked reply with 'This question is not relevant to my scope'
  Few good examples of python code output between #### seperator:
  ####
  {example}
  ####
  Code generator is shared below, delimited with triple backticks:
  ```
  {question}
  ```

'''

def generator(question,example):
    prompt_template = PromptTemplate.from_template(prompter)
    prompt = prompt_template.format(question = question,example = example)
    llm = Ollama(model = 'llama3',temperature=0.3)
    response = llm(prompt)
    return response

submit = st.button("Generate")
if submit:
    with st.spinner("Generating your code...."):
        st.write(generator(question,example))