
## Libraries
import os
from constants import openai_key
from langchain_openai import OpenAI

from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain

from langchain.chains.sequential import SequentialChain
from langchain.memory.buffer import ConversationBufferMemory

import streamlit as st

## OpenAI key
os.environ["OPENAI_API_KEY"] = openai_key

## UI
st.title("Langchain demo with Open AI")
input_text = st.text_input("Enter any celebrity person name")

## Open AI model
llm_model = OpenAI( model = "gpt-3.5-turbo-16k-0613", temperature=0.7)


## Prompt,memory and chains
first_prompt = PromptTemplate(input_variables=['name'],
                              template="Tell me about the celebrity {name}"
                              )
person_memory = ConversationBufferMemory(input_key='name',memory_key='person_history')

chain = LLMChain(llm=llm_model,prompt=first_prompt,verbose=True,output_key="person",memory=person_memory)


second_prompt = PromptTemplate(input_variables=['person'],
                               template = "when was {person} born")

dob_memory = ConversationBufferMemory(input_key='person',memory_key='person_history')

chain2 = LLMChain(llm=llm_model, prompt=second_prompt, verbose=True,output_key='dob',memory=dob_memory)


third_prompt = PromptTemplate(input_variables=['dob'],
                              template="5 major events happened on {dob}")
events_memory = ConversationBufferMemory(input_key='dob',memory_key='events_history')
chain3 = LLMChain(llm=llm_model,prompt=third_prompt, verbose=True, output_key='events',memory=events_memory)


## Final chain
final_chain = SequentialChain(chains=[chain,chain2,chain3],
                              input_variables=['name'],
                              output_variables=['person','dob','events'],
                              verbose=True)


if input_text:
    st.write(final_chain({'name':input_text}))

    with st.expander('Person Name'): 
        st.info(person_memory.buffer)

