import os
from constants import openai_key
from langchain_openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
import streamlit as st

## openAI secret key
os.environ["OPENAI_API_KEY"] = openai_key


##stream ui
st.title("Langchain demo with Open AI")
input_text = st.text_input("Enter any celebrity person name")

## Prompt
first_prompt = PromptTemplate(input_variables=['name'],
                              template="Tell me about the celebrity {name}")

## Open AI model
llm_model = OpenAI( model = "gpt-3.5-turbo-16k-0613", temperature=0.7)

chain = LLMChain(llm=llm_model,prompt=first_prompt,verbose=True)


if input_text:
    st.write(chain(input_text))

