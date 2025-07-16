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
from langchain_chroma import Chroma


# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
DATA_PATH = "dataset"
# FILE_PATH = "dataset/ADV.xlsx"
FILE_PATH = "dataset/NCEN_Answers.xlsx"
CHROMA_PATH = "chroma"
vector_db = Chroma(collection_name="ADV_File")


def main():
    # load excel
    doc = load_excel()

    # load each row into document chunks
    documents = toChunks(doc)

    add_to_chroma(documents)

    print("tester")
    return

def load_excel():
    docs =[]
    df = pd.read_excel(FILE_PATH)
    df = df.fillna("")

    return df

def toChunks(doc):
    chunks = []
    for i, row in doc.iterrows():
        

        row_text = "\n".join([f"{col}: {row[col]}" for col in doc.columns])
        # print(row_text)
        # print(i)
        metadatas = {
            "id": str(i+1),
            "page" : i+1,
            "source" : "excel"
        }

        chunk = Document(
            page_content = row_text,
            metadata = metadatas
        )
        chunks.append(chunk)
        # chunks.append(row_text)
        # print(i)
        # print(row_text)


    return chunks

# this loads currently data from a markdown file
# before loading this data we need to split it
# def load_excel_doc():    
#     df = pd.read_excel(FILE_PATH)
#     # print(df)

#     docs = []
#     for index, row in df.iterrows():
#         document = Document(
#             page_content=row['Question'],
#             meta_data ={'Section':'Section', 'Subsection':'Subsection', 'id':index},
#             id=str(index)
#         )
        
#         docs.append(document)

#     return docs

def add_to_chroma(chunks: list[Document]):
    # Load the existing database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Calculate Page IDs
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)


    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

    else:
        print("No new documents to add")
    return 

# FOR PDF - since documents are massive, we need to split the data in chunks
# def split_documents(documents: list[Document]):


#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=150,
#         chunk_overlap=20,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     # split the text into a list of chunks
#     chunks = text_splitter.split_documents(documents)
#     return chunks


def calculate_chunk_ids(chunks):
    # this will create IDs such as "dataset/Report Number 1 Fake Company.docx"
    # Page Source : Page Number : Chunk Index
    latest_page = None
    current_chunk = 0

    for chunk in chunks:
        # print("chunk")

        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index
        if current_page == latest_page:
            current_chunk += 1
        else:
            current_chunk = 0
        
        # Calculate the chunk ID
        chunk_id = f"{current_page}:{current_chunk}"
        latest_page = current_page


    return chunks


# After loading the doucments and splitting them into chunks, place it into a vector DB


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()