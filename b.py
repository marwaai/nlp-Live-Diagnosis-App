import streamlit as st
import os
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
import torch
class MyCustomHandler:
    def __init__(self, st_placeholder):
        self.st_placeholder = st_placeholder
        self.text = ""

    def on_llm_new_token(self, token: str):
        self.text += token + " "
        self.st_placeholder.text(self.text)

# Initialize persist directory
persist_directory = "db"

# Load documents and create embeddings
texts = []
for root, dirs, files in os.walk(r"db"):
    for file in files:
        if file.endswith("txt"):
            loader = TextLoader(os.path.join(root, file))
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            texts.extend(text_splitter.split_documents(documents))

# Create embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
db = FAISS.from_documents(texts, embeddings)

# Initialize GPT-2 pipeline
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def stream_output(question, st_placeholder):
    callback_handler = MyCustomHandler(st_placeholder)
    gpt2_pipeline.tokenizer = tokenizer
    generated_text = gpt2_pipeline(question, max_length=150, num_return_sequences=1)
    for output in generated_text:
        callback_handler.on_llm_new_token(output['generated_text'])

# Main Streamlit app logic
def main():
    st.title('Live Diagnosis App')
    st.write('Ask your medical questions here and get live answers from our AI assistant.')

    # Text input field for user to ask questions
    question = st.text_input('Enter your question:')
    placeholder = st.empty()
    if st.button('Ask'):
        stream_output(question, placeholder)

# Start the Streamlit app
if __name__ == '__main__':
    main()
