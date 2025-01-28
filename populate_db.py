import streamlit as st
import pandas as pd
import numpy as np

import argparse
import os
import shutil


from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma


# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
DATA_PATH = "dataset"


def main():
    # clear db
    # load documents
    doc = load_documents()
    # split documents
    chunks = split_documents(doc)
    # add chunks to chroma
    print("test")
    return


# this loads currently data from a markdown file
# before loading this data we need to split it
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.docx")
    documents = loader.load()
    # documents are just metadata that intakes the entire file
    return documents

# since documents are massive, we need to split the data in chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    # split the text into a list of chunks
    chunks = text_splitter.split_documents(documents)
    # how many chunks are in the document
    print(f"split {len(documents)} documents into {len(chunks)} chunks.")

    print(chunks[2].page_content)
    print(chunks[2].metadata)

    return chunks

# After loading the doucments and splitting them into chunks, place it into a vector DB




# st.write("Test Page")


if __name__ == "__main__":
    main()