# Bring in deps
import os 
from api_key import api_key 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredPDFLoader


os.environ['OPENAI_API_KEY'] = api_key

# App framework
st.title('⚖️ Legal Chatbot')
prompt = st.text_input('Enter your query/phrase/keyword') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'pdf_content'], 
    # template='Search for {title} within this content:{pdf_content} and provide relevant details in detail with all rules mentioned in content'
    # template='Provide a detailed explanation, using bullet points, about the rights I have regarding {title}. Leverage only this content:{pdf_content} '
    template = """List out all the laws in legal terms of the specific rights a customer possess in relation to \'{title}\',
    Use bullet points as a line, if necessary, by seraching within this content only : \"{pdf_content}\", 
    Give a brief summary as a new paragraph.
    
    """
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.1, max_tokens=250, max_retries=1) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

pdf_content = UnstructuredPDFLoader('./Consumer Rights UK.pdf')


# Show stuff to the screen if there's a prompt
if prompt: 
    # title = title_chain.run(prompt)
    title = prompt
    script = script_chain.run(title=title, pdf_content=pdf_content)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)
