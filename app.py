from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from flask import Flask, request, jsonify,render_template
# from pinecnone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.agents import create_agent
from dotenv import load_dotenv
import requests

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

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



@tool
def tavily_search(query: str) -> str:
    """
    ## Search the web using Tavily and return the results.
    Use this tool to search the web for information related to the user's question. Return the search results as a string.
    Give best result which is most relevant to the query. If no results are found, return "No results found."
    Args:
        query (str): The search query.
    Returns:
        str: The search results.
    """
    url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
    }
    params = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 3  # Tells the API to only fetch up to 3 results
    }
    
    try:
        response = requests.post(url, headers=headers, json=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                return "No results found."
                
            # Take the first 3 results and format them into a single string
            formatted_results = []
            for item in results[:3]:
                title = item.get("title", "No Title")
                url_link = item.get("url", "No URL")
                content = item.get("content", "")
                
                formatted_results.append(f"Title: {title}\nURL: {url_link}\nContent: {content}\n---")
                
            return "\n".join(formatted_results)
            
        return f"Error: API returned status code {response.status_code}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="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"
)

agent = create_agent(
    llm = llm,
    tools = [tavily_search],
    agent_type = "zero-shot-react-description",
    verbose = True
)






app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__  == "__main__":
    app.run(debug=True)
