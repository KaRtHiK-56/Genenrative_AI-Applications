import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Tuner", page_icon="🤖")
with st.sidebar:
    
    """The code snippet `st.text("PROMPT TUNER")` is displaying the text "PROMPT TUNER" in the
    Streamlit sidebar. This text serves as a heading or label to provide information or context to
    the user about the purpose or content of the section related to prompt tuning."""
    
    st.text("PROMPT TUNER")
    st.image(r"C:\Users\Devadarsan\Desktop\Karthik_projects\Prompt_Tuner\tuner.jpg")

st.title("🦜 LLM Prompt Tuner 🛠️ ")
st.text("Prompt tuner helps in tuning your propmt for specific usecases.")
text = st.text_area("Please enter your prompt here:")
configuration = st.selectbox(
    "Please select the configuration of the prompt to be designed:",
    ("Proper Wordings", "Accuracy", "Improved performance"),
)


def prompt(text, configuration):
   """
   The `prompt` function takes user input text and configuration, refines the prompt using a specified
   style, and generates a response using a language model.
   
   :param text: The `text` parameter in the `prompt` function represents the manually entered user
   prompt that needs to be refined and transformed using the specified configuration style. This text
   serves as the input for the function to generate a more effective prompt.

   :param configuration: The `configuration` parameter in the `prompt` function is used to specify the
   style or format that should be applied to transform the manually entered user prompt. This style
   will guide the process of refining and enhancing the prompt to make it more effective and clear for
   the user.

   :return: The `prompt` function is returning a response generated by a language model based on the
   refined prompt template provided in the function. The response is generated by passing the refined
   prompt to the language model for further processing and refinement.
   """
   
   print("The user input is ", text)
   print("The configuration is ", configuration)
   prompt = """
**Objective:**
Transform the manually entered user prompt {text} into a more refined and effective prompt using {configuration} style.

**Instructions:**

1. **Analyze the User Prompt {text}:**
   - Review the manually entered user prompt{text} using {configuration} style.

2. **Extract Core Intent:**
   - Determine the main purpose or goal of the prompt. What is the user seeking to achieve or learn?

3. **Identify Context and Details:**
   - Note any relevant background information or specific details that should be included to provide clarity.

4. **Clarify and Specify:**
   - Adjust the prompt to remove any ambiguities. Be as specific as possible to focus on the desired outcome.

5. **Enhance Precision and Language:**
   - Refine the wording to make the prompt clear, concise, and aligned with the intended purpose.

6. **Formulate the Refined Prompt:**
   - Combine the clarified intent, context, and precise language into a polished prompt.

---

**Template Example:**

**User Input:**
- [Enter the manually entered user prompt here]

**Refined Prompt:**
1. **Core Intent:** [Describe the main goal or question]
2. **Context:** [Include relevant background or specifics]
3. **Desired Output:** [Specify the expected type of response]
4. **Clarification:** [Eliminate any vague terms or ambiguities]

**Final Refined Prompt:**

**Refinement:** [Adjust the wording for clarity and precision]

"""

   llm = Ollama(model="llama3", temperature=0)
   prompt = PromptTemplate.from_template(template=prompt)
   prompt = prompt.format(text=text, configuration=configuration)
   response = llm(prompt)
   return response


if text is not None:
    submit = st.button("Tune Prompt")

    if submit:
      st.write(prompt(text, configuration))