from dotenv import load_dotenv
load_dotenv()
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import streamlit as st 
from langgraph.checkpoint.memory import MemorySaver

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="centered"
)

# ---------------- Header ---------------- #

st.title("🎯 AI Interview Coach")
st.caption("Practice technical, HR, and behavioral interviews with AI.")

st.divider()

# ---------------- Sidebar ---------------- #

with st.sidebar:
    st.header("Interview Settings")

    st.info(
        """
This AI helps you prepare for interviews.

It can generate:
- HR Questions
- Technical Questions
- Behavioral Questions
- Coding Questions
"""
    )

    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


llm=ChatGroq(model="openai/gpt-oss-20b")
search= GoogleSerperAPIWrapper()
memory=MemorySaver()
@st.cache_resource
def agentss():
    agnt=create_agent(model=llm,
                 tools=[search.run],
                 system_prompt="you are a ai interview prepare assistent you take user input like company job role experience and skill then you generate a proper interview question  ask the question one by one then give the proper interviws assistent and question remeber just ask company job role experience and type of interview question ",
                 checkpointer=memory
                 )
    return agnt
agent=agentss()

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt=st.chat_input("Tell me how can i help you ")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("User"):
        st.markdown(prompt)
    
    response = agent.invoke(
        {"messages": [{"role": "user","content": prompt}]},
        {"configurable": {"thread_id": "interview-chat"}})
    
    answer = response["messages"][-1].content
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)