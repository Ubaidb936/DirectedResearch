import shutil
import requests
from urllib.parse import urlparse
import sys
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from langchain_core.language_models import BaseChatModel
import json
import datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models import ChatHuggingFace
import os
import random
import time
from datasets import Dataset, DatasetDict


loader = PyPDFLoader("finance1.pdf")



text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  
        chunk_overlap=30,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
try:
    langchain_docs = loader.load_and_split(text_splitter=text_splitter) #loads and slits
    #docs = loader.load()
    #langchain_docs = text_splitter.split_documents(docs)
except Exception as e:
    print("An error occurred:", e)





from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

db = FAISS.from_documents(langchain_docs,
                          HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))






























