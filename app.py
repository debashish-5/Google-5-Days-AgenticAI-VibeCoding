from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from flask import Flask, request, jsonify,render_template
# from pinecnone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

prompt = PromptTemplate.from_template(  
    """
    ## A Professional Experienced AgenticAi/AI Teacher
    You are Teacher from Google 5 Days VibeCodes. You are a helpful assistant that helps students learn coding and programming concepts. You will answer the questions based on the context provided. 
    - Ask the user for clarification if the question is ambiguous or unclear.
    - Provide clear and concise explanations.
    - Use examples and analogies to help explain complex concepts.
    - Encourage the user to ask follow-up questions if they need further clarification.
    - Retrieve relevant information from the context to support your answers.
    - Your answers should be accurate, informative, and easy to understand related to retrieved context.

    ## Use this chat_history to answer the question. If the context is not enough, you can answer based on your knowledge. If you don't know the answer, just say "I don't know". Do not try to make up an answer.
    {chat_history}

    ## User Question: {question}
    ## Assistant:"""

)

from pinecone import Pinecone
import os
api_key = "pcsk_3Z18nd_New68WcRtj3Uybw1EaFKUvrPfg8Tp2AtbcD8Gdovfv6kPsnM7kkzJqt1Dqdaec8"

os.environ['PINECONE_API_KEY'] = api_key

def pinecone_retrive(query):
# Initialize the Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key="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"
    )


    # To this:
    index_name = "google-5days-vibecoding"


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pinecone_db = PineconeVectorStore(
        pinecone_api_key=api_key,
        index_name=index_name,
        embedding=embeddings,
        # embeddding_dimension=384
    )
    retriever = pinecone_db.as_retriever(search_kwargs={"k": 3})

    chat_history = []

    qa = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        return_source_documents=True,
        condense_question_prompt=prompt,
    )
    return qa.invoke({"question":query,"chat_history":chat_history})


from langchain_community.vectorstores import FAISS
##HNSW
def hnsw_retrive(query):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key="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"
    
    )
    # 1. Initialize the exact same embedding model used during saving
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 2. Load the index from your local folder
    # Note: allow_dangerous_deserialization=True is REQUIRED for pickle loading
    db = FAISS.load_local(folder_path="HNSW_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    chat_history = []

    qa = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        return_source_documents=True,
        condense_question_prompt=prompt,
    )
    return qa.invoke({"question":query,"chat_history":chat_history})


## PQ
def pq_retrive(query):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key="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"
    
    )
    # 1. Initialize the exact same embedding model used during saving
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # 2. Load the index from your local folder
    # Note: allow_dangerous_deserialization=True is REQUIRED for pickle loading
    db = FAISS.load_local(folder_path="PQ_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    chat_history = []

    qa = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retriever,
        return_source_documents=True,
        condense_question_prompt=prompt,
    )
    return qa.invoke({"question":query,"chat_history":chat_history})




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__  == "__main__":
    app.run(debug=True)
