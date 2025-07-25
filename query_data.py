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

Based off of the above context, answer the question from section {section} {question}

"""



def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    # df = pd.read_excel(Q_PATH)

    # df["Answer"] = ""

    # for i, row in df.iterrows():
    #     query_text = row["Question"]
    #     section_text = row["Section"]

    #     # print(query_text)
    #     answer = query_rag(query_text, section_text)
    #     row["Answer"] = answer

    # df.to_excel("output/answer.xlsx", sheet_name="NCEN", index=False)
        
    return

def query_rag(query_text: str, section_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB and pulls k lines of context -> document variable: ID, metadata, page_content.
    results = db.similarity_search_with_score(query_text, k=3)

    # Prepare prompt - Context Text = results but seperated.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    # Pull the prompt tempalte + add context + question.
    prompt_template = ChatPromptTemplate.from_template(FORM_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text, section=section_text)
    # print(prompt)

    # Pull model + ask model prompt.
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    # Prepare formatted response.
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()