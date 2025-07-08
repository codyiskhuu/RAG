# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain_ollama import OllamaEmbeddings

## This is where I would have to get an aws account

def get_embedding_function():
    # embeddings = ChatBedrockConverse(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings