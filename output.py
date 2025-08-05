import time

import streamlit as st
import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

import pandas as pd
import numpy as np

from get_embedding_function import get_embedding_function

# CHROMA_PATH = "chroma"
CHROMA_PATH = "form"

# Form input for questions
Q_PATH = "dataset/ncen_question.xlsx"


# Original Prompt

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# SEC Form Prompt

FORM_TEMPLATE = """
Answer the SEC Question based only on the following context:

{context}

---

Based off of the above context, answer the question: {question} from section {section}

Instructions: Provide an only the answer, no other words, and use only the provided context to answer the question.

"""
#  from section {section}


def main():
    # st.set_page_config(page_title="Chat with AI", page_icon="🤖")
    # st.title("🤖 Chat with AI")
    
    # # Initialize chat history
    # if "messages" not in st.session_state:
    #     st.session_state.messages = []
    # if "step" not in st.session_state:
    #     st.session_state.step = "ask_q"   
    # if "first_q" not in st.session_state:
    #     st.session_state.first_q = ""


    # # Display all previous messages
    # for msg in st.session_state.messages:
    #     with st.chat_message(msg["role"]):
    #         st.markdown(msg["content"])

    # # Chat input box
    # prompt = st.chat_input("Ask me anything...", key="chat_q")

    # if prompt:
    #     if st.session_state.step == "ask_q":
    #         st.session_state.step_2 = prompt
    #         st.session_state.messages.append({"role": "user", "content": prompt})
            
    #         with st.chat_message("user"):
    #             st.markdown(prompt)
            
    #         with st.chat_message("assistant"):
    #             # msg = "From which section?"
    #             # st.markdown(msg)
    #             with st.spinner("Thinking..."):
    #                 msg = query_rag(prompt, "test")
    #                 st.markdown(msg)
                    
                    
    #         st.session_state.messages.append({"role": "assistant", "content": msg})    
            
    #         st.session_state.step  = "ask_q"



    # Terminal intake commands
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text

    # Data File input
    df = pd.read_excel(Q_PATH)

    df["Answer"] = ""
    df["Context"] = ""

    for i, row in df.iterrows():
        start = time.time()
        query_text = str(row["Question"]).strip()
        section_text = str(row["Section"]).strip()

        # print(query_text)
        answer, context= query_rag(query_text, section_text)
        # row["Answer"] = answer
        # row["Context"] = context
        df.at[i, "Answer"] = answer.strip()
        df.at[i, "Context"] = context.strip()
        time.sleep(1.5)
        end = time.time()
        print(f"Execution time: {end - start:.4f} seconds")

        
    print(df["Answer"])

    df.to_excel("output/answer_v1.xlsx", sheet_name="NCEN", index=False)
        
    return

def query_rag(query_text: str, section_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB and pulls k lines of context -> document variable: ID, metadata, page_content.
    results = db.similarity_search_with_score(str(query_text + " from section: " + section_text), k=3)

    # Prepare prompt - Context Text = results but seperated.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # Pull the prompt tempalte + add context + question.
    prompt_template = ChatPromptTemplate.from_template(FORM_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query_text, section=section_text)
    prompt = prompt_template.format(context=context_text, question=query_text, section=section_text)    
    # print(prompt)

    # Pull model + ask model prompt.
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    # Prepare formatted response.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(query_text)
    print(formatted_response)
    return response_text, context_text


if __name__ == "__main__":
    main()