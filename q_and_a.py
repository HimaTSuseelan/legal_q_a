import os
import openai
from api_key import api_key
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredPDFLoader

openai.api_key = api_key

# App framework
st.title('⚖️ Consumer Rights Q & A')
st.write('Ask a question and get an answer from the specified PDF file.')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
query_template = PromptTemplate(
    input_variables=['topic'],
    template='Details related to {topic}'
)

consumer_acts_template = PromptTemplate(input_variables=['query', 'pdf_content'],
    template='Act like a litigator and refer the document :{pdf_content} and display the section of {query} and the rules related to that')

# consumer_acts_template = PromptTemplate(
#     input_variables=['query', 'pdf_content'],
#     template='User: What are my rights as a consumer in the UK regarding {query}? I have a PDF that contains relevant information: {pdf_content}. Which section covers these rights?'
# )


# Memory
query_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
consumer_acts_memory = ConversationBufferMemory(input_key='query', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.01, model_name='gpt-3.5-turbo', max_tokens=1024, max_retries=1)
query_chain = LLMChain(llm=llm, prompt=query_template, verbose=True, output_key='query', memory=query_memory)
consumer_acts_chain = LLMChain(llm=llm, prompt=consumer_acts_template, verbose=True, output_key='consumer_acts', memory=consumer_acts_memory)

pdf_extractor = UnstructuredPDFLoader('./Consumer Rights UK.pdf')

# Show stuff to the screen if there's a prompt
if prompt:
    query = query_chain.run(prompt)
    pdf_content = pdf_extractor.load()
    consumer_acts = consumer_acts_chain.run(query=query, pdf_content=pdf_content)

    st.write(query)
    st.write(consumer_acts)

    with st.expander("Query Memory"):
        st.info(query_memory.buffer)

    with st.expander("Consumer acts"):
        st.info(consumer_acts_memory.buffer)
