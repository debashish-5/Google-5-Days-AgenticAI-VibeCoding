from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from flask import Flask, request, jsonify,render_template
# from pinecnone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_agent
from dotenv import load_dotenv
from openai import embeddings
import requests
import os
from langsmith import traceable
import numpy as np
#keyword search
from langchain_community.retrievers import BM25Retriever
#Hybrid Search
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import os
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool


splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

docs = PyPDFDirectoryLoader('documents/')
documents = docs.load()
embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

chunks = splitter.split_documents(documents)
faiss_db = FAISS.from_documents(chunks,embeddings)
faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 3}) ## Top 3 relevant documents will be retrieved for each query
#bm25 retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3  ## Top 3 relevant documents will be retrieved for each query

#hybrid search
hybrid_retriever = EnsembleRetriever(
    retrievers= [
        bm25_retriever,
        faiss_retriever
    ],
    weights = [0.4,0.6]
)



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


api_key = os.getenv("PINECONE_API_KEY")

os.environ['PINECONE_API_KEY'] = api_key

def pinecone_retrive(query):
# Initialize the Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
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
        "api_key": os.getenv("TAVILY_API_KEY"),
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
class retriever_x_llm:
    def __init__(self,llm = None, embeddings = None):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        
        )
        # 1. Initialize the exact same embedding model used during saving
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.condense_question_prompt = """
            ## A Professional Experienced AgenticAi/AI Teacher
            You are Teacher from Google 5 Days VibeCodes. You are a helpful assistant that helps students learn coding and programming concepts. You will answer the questions based on the context provided.
            - Ask the user for clarification if the question is ambiguous or unclear.
            - Provide clear and concise explanations.
            - Use examples and analogies to help explain complex concepts.
            - Encourage the user to ask follow-up questions if they need further clarification.
            - Retrieve relevant information from the context to support your answers.
            - Your answers should be accurate, informative, and easy to understand related to retrieved context.
            - You can use tool for better more accurate result.
            """
    ##HNSW
    def hnsw_retrive(self,query):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        
        )
        # 1. Initialize the exact same embedding model used during saving
        # use self.embeddings for consistency
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # 2. Load the index from your local folder
        # Note: allow_dangerous_deserialization=True is REQUIRED for pickle loading
        db = FAISS.load_local(folder_path="HNSW_faiss", embeddings=self.embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # chat_history = []

        # qa = ConversationalRetrievalChain.from_llm(
        #     llm = llm,
        #     retriever = retriever,
        #     return_source_documents=True,
        #     condense_question_prompt=prompt,
        # )
        # return qa.invoke({"question":query,"chat_history":chat_history})
        brainer = create_retriever_tool(
            retriever,
            name="HNSW Retriver",
            description="Searches internal documents. Use this as your primary source for internal info."
        )
        agent = create_agent(
            model = llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt
        )

        return agent.invoke({'messages':query})


    ## PQ
    def pq_retrive(self,query):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        
        )
        # 1. Initialize the exact same embedding model used during saving
        # use self.embeddings initialized in constructor
        # 2. Load the index from your local folder
        # Note: allow_dangerous_deserialization=True is REQUIRED for pickle loading
        db = FAISS.load_local(folder_path="PQ_faiss", embeddings=self.embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # chat_history = []

        # qa = ConversationalRetrievalChain.from_llm(
        #     llm = llm,
        #     retriever = retriever,
        #     return_source_documents=True,
        #     condense_question_prompt=prompt,
        # )
        brainer = create_retriever_tool(
            retriever,
            name = "PQ_faiss Retriever",
            description="Searches internal documents. Use this as your primary source for internal info."

        )
        agent = create_agent(
            model = self.llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt

        )
        return agent.invoke({'messages':query})

    def ivf_retriever(self,query):
        db = FAISS.load_local(folder_path="IVF_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        brainer = create_retriever_tool(
            retriever,
            name = "IVF_faiss Retriever",
            description="Searches internal documents. Use this as your primary source for internal info."

        )
        agent = create_agent(
            model = self.llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt

        )
        return agent.invoke({'messages':query})
        

    def ivfpq_retriever(self,query):
        db = FAISS.load_local(folder_path="IVFPQ_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        chat_history = []
        brainer = create_retriever_tool(
            retriever,
            name = "IVFPQ_faiss Retriever",
            description="Searches internal documents. Use this as your primary source for internal info."

        )
        agent = create_agent(
            model = self.llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt

        )
        return agent.invoke({'messages':query})
        
        
    
    def hybrid_search(self,query):
        chat_history = []
        qa = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever = hybrid_retriever,
            return_source_documents=True,
            condense_question_prompt=self.condense_question_prompt,
        )
        try:
            return qa.invoke({"question":query,"chat_history":chat_history})
        except Exception as e:
            return f"An error occurred during hybrid search: {str(e)}"
        







class retriever:
    def __init__(self,llm = None, embeddings = None):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        
        )
        # 1. Initialize the exact same embedding model used during saving
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ##HNSW
    def hnsw_retrive(query):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        
        )
        # 1. Initialize the exact same embedding model used during saving
        # use self.embeddings for consistency
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # 2. Load the index from your local folder
        # Note: allow_dangerous_deserialization=True is REQUIRED for pickle loading
        db = FAISS.load_local(folder_path="HNSW_faiss", embeddings=self.embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # chat_history = []

        # qa = ConversationalRetrievalChain.from_llm(
        #     llm = llm,
        #     retriever = retriever,
        #     return_source_documents=True,
        #     condense_question_prompt=prompt,
        # )
        # return qa.invoke({"question":query,"chat_history":chat_history})
        return retriever.invoke(query)



    ## PQ
    def pq_retrive(self,query):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        
        )
        # 1. Initialize the exact same embedding model used during saving
        # use self.embeddings initialized in constructor
        # 2. Load the index from your local folder
        # Note: allow_dangerous_deserialization=True is REQUIRED for pickle loading
        db = FAISS.load_local(folder_path="PQ_index", embeddings=self.embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # chat_history = []

        # qa = ConversationalRetrievalChain.from_llm(
        #     llm = llm,
        #     retriever = retriever,
        #     return_source_documents=True,
        #     condense_question_prompt=prompt,
        # )
        return retriever.invoke(query)
    def ivf_retriever(self,query):
        db = FAISS.load_local(folder_path="IVF_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriver = db.as_retriever(search_kwargs={"k": 3})

        return retriver.invoke(query)

    def ivfpq_retriever(self,query):
        db = FAISS.load_local(folder_path="IVF_PQ_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        return retriever.invoke(query)
        





llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

agent = create_agent(
    model = llm,
    tools = [tavily_search]
)






app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


retriever_llm = retriever_x_llm()
retriever = retriever()
@app.route('/chat', methods=['POST'])
def chat():


    data = request.get_json()
    user_query = data.get("message")  # Get the user query from the request
    faiss_model = data.get("vector_store")  # Get the selected vector store from the request
    if not user_query or not faiss_model:
        return jsonify({"error": "Missing user query or vector store selection"}), 400
    try:
        if faiss_model == "faiss-hnsw":
            result = retriever_llm.hnsw_retrive(user_query)
        elif faiss_model == "faiss-pq":
            result = retriever_llm.pq_retrive(user_query)
        else:
            return jsonify({"error": "Unknown vector store selection"}), 400

        answer = result['messages'][-1].content
        return jsonify({"response": answer})  # Return the response as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__  == "__main__":
    app.run(debug=True)
