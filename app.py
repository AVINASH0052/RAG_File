import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional, Dict
import requests
import json
import PyPDF2
import docx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import Field, model_validator
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Document Chat", page_icon="📚")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

class NvidiaLLM(LLM):
    """Custom LLM class for NVIDIA API"""
    
    client: Any = Field(default=None, exclude=True)
    model: str = Field(default="nvidia/llama-3.1-nemotron-ultra-253b-v1")
    temperature: float = Field(default=0.6)
    api_key: str = Field(default="")
    
    @model_validator(mode='before')
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that api key exists in environment."""
        api_key = values.get("api_key")
        if not api_key:
            raise ValueError("api_key must be provided")
        
        values["client"] = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        return values
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=4096,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        return completion.choices[0].message.content
    
    @property
    def _llm_type(self) -> str:
        return "nvidia"

def process_documents(documents):
    """Process uploaded documents and return text content."""
    all_texts = []
    for doc in documents:
        try:
            if doc.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(doc)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                all_texts.append(f"Document: {doc.name}\n{text}")
            elif doc.name.endswith('.docx'):
                doc = docx.Document(doc)
                text = ""
                for para in doc.paragraphs:
                    text += para.text + "\n"
                all_texts.append(f"Document: {doc.name}\n{text}")
            elif doc.name.endswith('.txt'):
                text = doc.getvalue().decode()
                all_texts.append(f"Document: {doc.name}\n{text}")
        except Exception as e:
            st.error(f"Error processing {doc.name}: {str(e)}")
            continue
    
    return "\n\n".join(all_texts)

def create_vector_store(text):
    """Create vector store from text."""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings using a more stable model
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": "cpu"}
    )
    
    # Create vector store using FAISS
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    
    return vector_store

def create_conversation_chain(vector_store):
    """Create conversation chain."""
    llm = NvidiaLLM(
        api_key="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"
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Custom prompt template for better summarization
    template = """You are a helpful AI assistant that provides clear and concise answers based on the provided context.
    When asked for a summary, provide a comprehensive overview of all documents.
    For specific questions, focus on answering directly without unnecessary document references.
    Always prioritize accuracy and relevance in your responses.
    
    Context: {context}
    
    Question: {question}
    
    Provide a clear and focused response:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8}  # Increased number of chunks for better context
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return conversation_chain

# Main UI
st.title("📚 Document Chat")
st.write("Upload your documents and chat with them!")

# File uploader
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT)",
    type=['pdf', 'docx', 'txt'],
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Maximum 5 files allowed. Only the first 5 files will be processed.")
        uploaded_files = uploaded_files[:5]
    
    # Process documents
    with st.spinner("Processing documents..."):
        raw_text = process_documents(uploaded_files)
        vector_store = create_vector_store(raw_text)
        st.session_state.vector_store = vector_store
        st.session_state.conversation = create_conversation_chain(vector_store)
    st.success(f"Successfully processed {len(uploaded_files)} document(s)!")

# Chat interface
if st.session_state.conversation is not None:
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation.invoke({
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })
            st.session_state.chat_history.append((user_question, response["answer"]))
        
        # Display chat history
        for question, answer in st.session_state.chat_history:
            st.write("Q:", question)
            st.write("A:", answer)
            st.write("---") 
