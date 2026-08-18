from chromadb import Embeddings
from envs.langgraph_env.Lib import re
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from io import BytesIO
import glob
from langsmith import trace, traceable
import numpy as np
#keyword search
from langchain_community.retrievers import BM25Retriever
#Hybrid Search
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import os
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langsmith import traceable 
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
#for multiquery retriver 
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter,LLMChainExtractor,LLMChainFilter,DocumentCompressorPipeline
#selfquery retriever
from langchain_classic.chains.self_query_retriever import SelfQueryRetriever
#parent child retriever
from langchain_classic.retrievers import ParentChildRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_classic.storage import InMemoryStore
import faiss
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
# autogen integration 
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import TextMentionTermination,HandoffTermination,FunctionalTermination,TextMessageTermination
load_dotenv()
# autogen
model_client = OpenAIChatCompletionClient(model="llama-3.1-8b-instant",
                                          base_url = "https://api.groq.com/v1/",
                                          model_info = {
                                              'vision':True,
                                              'function_calling':True,
                                              'json_output':True,
                                              'structured_output':True,
                                              'family':"unknown"
                                          })



docs = PyPDFDirectoryLoader("documents")
documents = docs.load()
parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000,chunk_overlap = 200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size = 400,chunk_overlap = 50)


#-------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

documents = []
for pdf_path in sorted(glob.glob(os.path.join("documents", "*.pdf"))):
    if os.path.basename(pdf_path).startswith("~$"):
        continue
    try:
        documents.extend(PyPDFLoader(pdf_path).load())
    except Exception as exc:
        print(f"Skipping unreadable PDF '{pdf_path}': {exc}")

Field_embeddings = HuggingFaceBgeEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

chunks = splitter.split_documents(documents)
faiss_db = FAISS.from_documents(chunks,Field_embeddings)
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
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

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



@traceable(name = "pinecone_retriever")
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
@traceable(run_type="tool")
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
    @traceable(name="init_setup")
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
    @traceable(name="hnsw_retriever_agent")
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
            name="HNSW_Retriever",
            description="Searches internal documents. Use this as your primary source for internal info."
        )
        agent = create_agent(
            model = llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt
        )

        return agent.invoke({"messages": [HumanMessage(content=query)]})


    ## PQ
    @traceable(name="pq_retriever_agent")
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
            name = "PQ_faiss_Retriever",
            description="Searches internal documents. Use this as your primary source for internal info."

        )
        agent = create_agent(
            model = self.llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt

        )
        return agent.invoke({"messages": [HumanMessage(content=query)]})
    @traceable(name = "ivf_retriever_agent")
    def ivf_retriever(self,query):
        db = FAISS.load_local(folder_path="IVF_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        brainer = create_retriever_tool(
            retriever,
            name = "IVF_faiss_Retriever",
            description="Searches internal documents. Use this as your primary source for internal info."

        )
        agent = create_agent(
            model = self.llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt

        )
        return agent.invoke({"messages": [HumanMessage(content=query)]})
        
    @traceable(name = "ivfpd_retriever_agent")
    def ivfpq_retriever(self,query):
        db = FAISS.load_local(folder_path="IVFPQ_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        chat_history = []
        brainer = create_retriever_tool(
            retriever,
            name = "IVFPQ_faiss_Retriever",
            description="Searches internal documents. Use this as your primary source for internal info."

        )
        agent = create_agent(
            model = self.llm,
            tools=[brainer,tavily_search],
            system_prompt=self.condense_question_prompt

        )
        return agent.invoke({"messages": [HumanMessage(content=query)]})
        
        
    @traceable(name="hybrid_search")
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
        

class Deeper:
    @traceable(name = "DeepSearch_init")
    def __init__(self,llm = None):
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
    @traceable(name = "Deepth Level 1: Query Rewriting")
    def Lv1_query_rewriting(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs = {"k":3})

        rewrite_prompt = ChatPromptTemplate.from_template(
            """
            You are a search query optimizer.

            Rewrite the user's question into a better search query.

            Rules:
            - Keep the same meaning.
            - Expand abbreviations.
            - Replace pronouns with expplicit nouns.
            - Use technical terminology when possible.
            - Return ONLY the Rewritten query.

            Question:
            {question}
            """
        )
        # creating rewrite function
        @traceable(name = "rewrite_query")
        def rewrite_query(query):
            prompt = rewrite_prompt.invoke({'question': query})
            response = self.llm.invoke(prompt)
            return response.content.strip()
        ##final output retriever + llm
        docs = retriever.invoke(rewrite_prompt({'question': query}))
        context = "\n\n".join(doc.page_content for doc in docs)
        final_answer_prompt = ChatPromptTemplate.from_template(
            """
            Answer using only the context.
            
            Context:
            {context}

            Question:
            {question}
            """
        )
        chain = final_answer_prompt | self.llm

        response = chain.invoke({'context':context,'question':query})
        return response.content

    @traceable(name = "Deepth Level 2: Multiquery Retriever")
    def Lv2_multiquery_rewriting(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs = {'k':3})
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever= retriever,
            llm = self.llm
        )
        docs = multi_retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        final_answer_prompt = ChatPromptTemplate.from_template(
            """
            Answer using only the context.

            Context:
            {context}

            Question:
            {question}

            """
        )
        chain = final_answer_prompt | self.llm
        response = chain.invoke({'context':context,'question':query})
        return response.content

    @traceable(name = "HyDE: Hypothitical Documents Embeddings with LLM")
    def Lv3_HyDE_with_LLM(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs = {'k':3})
        #prompt
        hyde_prompt = ChatPromptTemplate.from_template(
            """
            Write a detailed paragraph that would likely appear
            inside a document answering the user's question.

            Do not say "I don't know."
            Write as if this paragraph exists inside a PDF.

            Question:
            {question}
            """
        )
        ### function for HyDE generation
        def generate_hypothetical_document(question):
            prompt = hyde_prompt.invoke({'question': question})
            response = self.llm.invoke(prompt)
            return response.content
        docs = generate_hypothetical_document(query)
        
        context = '\n\n'.join(doc.page_content for doc in docs)

        #complete the answer with llm
        answer_prompt = ChatPromptTemplate.from_template("""
        Context:
        {context}

        Question:
        {question}
        Answer only using the context.
        """)

        chain = answer_prompt | self.llm
        result = chain.invoke({'context':context,'question':query})

        return result.content 
    @traceable(name = "Query Decomposition")
    def Lv4_Query_Decomposition(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs = {'k':3})
        decompose_prompt = ChatPromptTemplate.from_template("""
        You are an expert at breaking complex questions into smaller independent search questions.

        Rules:
        - Generate between 3-6 search questions.
        - Each question should retrieve different information.
        - Avoid duplicates.
        - Return one question per line.
        - Do not include answers or context, only the questions.

        Question:
        {question}
        """)

    @traceable(name = "Parent Child Retrieval")
    def Lv5_Parent_Child_Retrieval(self,query,file_name):
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        vectorstore = FAISS(embeddings = Field_embeddings, index = index,docstore = InMemoryDocstore(), index_to_docstore_id = {})
        store = InMemoryStore()
        retriever = ParentChildRetriever(
            vectorstore = vectorstore,
            docstore = store,
            parent_splitter = parent_splitter,
            child_splitter = child_splitter
        )
        retriever.add_documents(documents)
        docs = retriever.invoke(query)
        context = '\n\n'.join(doc.page_content for doc in docs)
        prompt = ChatPromptTemplate.from_template(
            """
            Answer only using the context.
            Context:
            {context}
            Question:
            {question}
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({'context':context,'question':query})
        return result
    
    #decompose function
        def decompose_query(query):
            prompt = decompose_prompt.invoke({'question':query})
            response = self.llm.invoke(prompt)
            queries = response.content.strip().split('\n')
            return queries
        queries = decompose_query(query)
        all_docs = []
        for query in queries:
            docs = retriever.invoke(query)
            all_docs.extend(docs)

        #remove duplicates
        unique_docs = {}
        for doc in all_docs:
            key = (
                doc.metadata['source'],
                doc.metadata['page']
            )
            unique_docs[key] = doc
        docs = list(unique_docs.values())
        context = '\n\n'.join(doc.page_content for doc in docs if docs)
        answer_prompt = ChatPromptTemplate.from_templaye(
            """
            Answer only using the context.
            Context:
            {context}
            Question:
            {question}
            """
        )
        chain = answer_prompt | self.llm
        result = chain.invoke({'context':context,'question':query})
        return result.content
    @traceable(name = "EmbeddingFilterRetriever")
    def Lv6_EmbeddingFilterRetriever(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs = {'k':3})
        filter_emb = EmbeddingsFilter(
            embeddings = Field_embeddings,
            similarity_threshold = 0.7
        )
        compression_retriever = ContextualCompressionRetriever(
            base_retriever = retriever,
            compressor = filter_emb
        )
        docs = compression_retriever.invoke(query)
        context = '\n\n'.join(doc.page_content for doc in docs)
        answer_prompt = ChatPromptTemplate.from_template(
            """
            Answer only using the context.
            Context:
            {context}
            Question:
            {question}
            """
        )
        chain = answer_prompt | self.llm | StrOutputParser()
        result = chain.invoke({'context':context,'question':query})
        return result
    @traceable(name = "LLMChainExctor")
    def Lv7_LLMChainExtractor(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs=  {'k':5})
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=compressor
        )
        docs = compression_retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = ChatPromptTemplate.from_template(
            """
            Answer only using the context.

            Context:
            {context}

            Question:
            {question}
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({'context':context,'question':query})
        return result

    @traceable(name = "LLMChainFilter")
    def Lv8_LLMChainFilter(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs=  {'k':5})
        compressor = LLMChainFilter.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_retriever = retriever,
            base_compressor = compressor
        )
        docs = compression_retriever.invoke(query)
        context = '\n\n'.join(doc.page_content for doc in docs)
        prompt = ChatPromptTemplate.from_template(
            """
            Answer only using the context.

            Context:
            {context}

            Question:
            {question}
            """
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({'context':context,'question':query})
        return result

    @traceable(name = "DocumentCompressorPipeline")
    def Lv9_DocumentCompressorPipeline(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs = {'k':20})
        #embedding filter
        embedding_filter = EmbeddingsFilter(
            embeddings= Field_embeddings,
            similarity_threshold=0.70
        )
        #LLM filter
        llm_filter = LLMChainFilter.from_llm(self.llm)
        #LLM Extractor
        extractor = LLMChainExtractor.from_llm(self.llm)
        #create pipeline 
        pipeline = DocumentCompressorPipeline(
            transformers=[
                embedding_filter,
                llm_filter,
                extractor
            ]
        )
        #wrap retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline,
            base_retriever = retriever
        )
        docs = compression_retriever.invoke(query)
        context = '\n\n'.join(doc.page_content for doc in docs)
        prompt = PromptTemplate.from_template("""
        Answer the question using only the context provided.

        Context:
        {context}

        Question:
        {query}
        """)
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({'query':query})
        return result

    @traceable(name = "CrossEncoderReranker")
    def Lv10_CrossEncoderReranker(self,query,file_name):
        db = FAISS.load_local(file_name)
        retriever = db.as_retriever(search_kwargs = {'k':20})
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        docs = retriever.invoke(query)
        pairs = [[query,doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        # Sort documents by their reranked scores
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
        context = '\n\n'.join(doc.page_content for doc in docs)
        prompt = PromptTemplate.from_template("""
        Answer the question using only the context provided.
        Question:
        {query}

        Context:
        {context}

        """)
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({'query':query,'context':context})
        return result


class hardcore:
    @traceable(name = "Hardcore")
    def fullPower(self,query,file_name):
        deeper = Deeper()
        result_v1 = deeper.Lv1_query_rewriting(query,file_name)
        result_v2 = deeper.Lv2_multiquery_rewriting(result_v1,file_name)
        result_v3 = deeper.Lv3_HyDE_with_LLM(result_v2,file_name)
        result_v4 = deeper.Lv4_Query_Decomposition(result_v3,file_name)
        result_v5 = deeper.Lv5_Parent_Child_Retrieval(result_v4,file_name)
        result_v6 = deeper.Lv6_EmbeddingFilterRetriever(result_v5,file_name)
        result_v7 = deeper.Lv7_LLMChainExtractor(result_v6,file_name)
        result_v8 = deeper.Lv8_LLMChainFilter(result_v7,file_name)
        result_v9 = deeper.Lv9_DocumentCompressorPipeline(result_v8,file_name)
        result_v10 = deeper.Lv10_CrossEncoderReranker(result_v9,file_name)
        return result_v10


class retriever:
    def __init__(self,llm = None, embeddings = None):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        
        )
        # 1. Initialize the exact same embedding model used during saving
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ##HNSW

    @traceable(name = "hnsw")
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
        return retriever.invoke(query)



    ## PQ
    @traceable(name = "pq")
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
        return retriever.invoke(query)
    @traceable(name = "ivf")
    def ivf_retriever(self,query):
        db = FAISS.load_local(folder_path="IVF_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriver = db.as_retriever(search_kwargs={"k": 3})

        return retriver.invoke(query)

    @traceable(name= "ivfpq")
    def ivfpq_retriever(self,query):
        db = FAISS.load_local(folder_path="IVFPQ_faiss",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        return retriever.invoke(query)
        


class AgenticRAG:
    @traceable(name= "AgenticRAG_init")
    def __init__(self,llm = None,tools=None):
        self.llm = ChatGroq(
            model = 'llama-3.1-8b-instant',
            api_key = os.getenv("GROQ_API_KEY")
        )
        self.duckduckgosearch = DuckDuckGoSearchResults()
        self.basic_search
        self.hybrid_search
        self.context_compression
        self.duckduckgosearch
        

    @tool
    @traceable(run_type="tool")
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
    @tool
    def basic_search(query:str,file_name :str = None) -> str:
        """
        ## Search the Vector Store and return the results.
        Args:
            query (str): The search query.
        Returns:
            str: The search results.
        """
        try:
            if file_name:
                file = file_name
            else:
                file = "faiss_index"
            
            db = FAISS.load_local(file,allow_dangerous_deserialization = True)
            retriever = db.as_retriever(search_kwargs={'k':3})
            result = retriever.invoke(query)
            return '\n\n'.join(doc.page_content for doc in docs)
        except Exception as e:
            return 'Erorr Occurs during {e}.'

    @tool
    def hybrid_search(query:str,file_name:str = None) -> object:
        """ ## Hybrid Search with LLM"""
        try:
            if file_name:
                file = file_name
            else:
                file = "faiss_index"
            db = FAISS.load_local(file,allow_dangerous_deserialization = True)
            sementic_retriever = db.as_retriever(search_kwargs = {'k':3})
            bm25_retriever = BM25Retriever.from_documents(chunks)
            bm25_retriever.k= 3
            hybrid_retriever = EnsembleRetriever(retrievers=[sementic_retriever,bm25_retriever],weights=[0.6,0.4])
            hybrid_result = hybrid_retriever.invoke(query)
            context = '\n\n'.join(doc.page_content for doc in hybrid_result)
            return hybrid_result
        except Exception as e:
            return f"Error: Occurs in {e}"
    @tool
    def context_compression(query:str,file_name:str = None) -> str:
        """
            ## Context Compression Power
        aim: If user ask about Compressing then use this.
        Args:
            query: user input
            file_name: option (if user give filename then use this otherwise use default "faiss_index" file )
        return:
            return the compressed result 
            
        """
        if file_name:
            file = file_name
        file = "faiss_index"
        db = FAISS.load_local(file,allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs = {'k':10})
        filteer = EmbeddingsFilter(embeddings=Embeddings,similarity_threshold=0.75)
        #compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=filteer,
            base_retriever=retriever
        )
        result = compression_retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
        # langchain Agent
    @tool
    def Bigneer_Retriever_Agent(query:str) -> list:
        """# A Bignner Retriever Agent
        Args:
            query: User input.
        return: 
            Agent Answer
        """
        basic_retriver_agent = create_agent(llm,tools=[basic_search])
        return basic_retriver_agent.invoke({'messages':query})
    @tool
    def intermidate_retriever_agent(query:str) -> list:
        """# A intermidate Retriever Agent
        Args:
            query: User input.
        return: 
            Agent Answer
        """
        intermidate_agent = create_agent(model = llm,tools= [hybrid_retriever])
        return intermidate_agent.invoke({'messages':query})
    @tool
    def advanced_retriever_agent(query:str) -> list:
        """# A Advanced Retriever Agent
        Args:
            query: User input.
        return: 
            Agent Answer
        """
        advanced_agent = create_agent(model = llm,tools=[basic_search,hybrid_retriever,context_compression,tavily_search])
        return advanced_agent.invoke({'messages':query})

    ## Langchain agent 
    def run_agent(self,query):
        agent_x1 = create_agent(
            model = self.llm,
            tools = [self.Bigneer_Retriever_Agent,self.intermidate_retriever_agent,self.advanced_retriever_agent],
            system_prompt = "You are Proffesional AiAgent"
        )
        chain = agent_x1 | StrOutputParser()
        return chain.invoke({'messages':query})

    




    
        



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
@traceable(name = "home")
def home():
    return render_template('index.html')


retriever_llm = retriever_x_llm()
retriever = retriever()
@app.route('/chat', methods=['POST'])
@traceable(name = "chat_endpoint")
def chat():


    data = request.get_json()
    user_query = data.get("message")  # Get the user query from the request
    faiss_model = data.get("vector_store")  # Get the selected vector store from the request
    if not user_query or not faiss_model:
        return jsonify({"error": "Missing user query or vector store selection"}), 400
    try:
        if faiss_model == "faiss-hnsw":
            result = retriever_llm.hnsw_retrive(user_query)
            answer = result["messages"][-1].content
            sources = []
        elif faiss_model == "faiss-pq":
            result = retriever_llm.pq_retrive(user_query)
            answer = result["messages"][-1].content
            sources = []
        elif faiss_model == "faiss-ivf":
            result = retriever_llm.ivf_retriever(user_query)
            answer = result["messages"][-1].content
            sources = []
        elif faiss_model == "faiss-ivfpq":
            result = retriever_llm.ivfpq_retriever(user_query)
            answer = result["messages"][-1].content
            sources = []
        elif faiss_model == "pinecone":
            result = pinecone_retrive(user_query)
            answer = result.get("answer", "I couldn't generate an answer.")
            sources = [
                {
                    "title": doc.metadata.get("source", "Pinecone document"),
                    "snippet": (doc.page_content or "")[:400],
                    "score": 0.9,
                }
                for doc in result.get("source_documents", [])
            ]
        else:
            return jsonify({"error": "Unknown vector store selection"}), 400

        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__  == "__main__":
    app.run(debug=True)
