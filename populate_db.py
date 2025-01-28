import streamlit as st
import pandas as pd
import numpy as np

import argparse
import os
import shutil

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma


DATA_PATH = "ADD_FOLDER"


def main():

    # clear db
    # load documents
    # split documents
    # add chunks to chroma

    return




# this loads currently data from a markdown file
# before loading this data we need to split it
def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    # documents are just metadata that intakes the entire file
    return documents

# since documents are massive, we need to split the data in chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# st.write("Test Page")
