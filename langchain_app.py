import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the output schema using Pydantic
class RecipeRecommendation(BaseModel):
    recipe_name: str = Field(description="Name of the recipe")
    cooking_time: str = Field(description="Estimated cooking time")
    ingredients: List[str] = Field(description="List of main ingredients")
    instructions: Optional[str] = Field(description="Brief cooking instructions")
    difficulty: str = Field(description="Difficulty level (Easy, Medium, Hard)")
    
    class Config:
        schema_extra = {
            "example": {
                "recipe_name": "Spaghetti Carbonara",
                "cooking_time": "20 minutes",
                "ingredients": ["spaghetti", "eggs", "pecorino cheese", "guanciale", "black pepper"],
                "instructions": "Cook pasta, prepare sauce with eggs and cheese, mix with crispy guanciale",
                "difficulty": "Medium"
            }
        }

# Initialize embedding model
@st.cache_resource
def get_embeddings():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAIEmbeddings(
        model="text-embedding-3-small"  # Using the smaller, more cost-effective model
    )

# Initialize LLM
@st.cache_resource
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768",
        temperature=0.7,
    )

# Load existing vectorstore
@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    
    # Check if vectorstore exists
    if not os.path.exists("./chroma_db"):
        raise ValueError("Vectorstore not found. Please run generate_vectorstore.py first!")
    
    # Load the existing vectorstore
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    return vectorstore

# Create the conversational chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=RecipeRecommendation)
    
    # Create custom prompt template
    template = """You are a knowledgeable and friendly cooking assistant. Use the following pieces of context to 
    provide helpful recipe recommendations and cooking advice. If you recommend a recipe, format it according to 
    the specified JSON schema. Pay attention to the metadata which includes cooking time and serving size.

    Context: {context}
    
    Chat History: {chat_history}
    
    Human: {question}
    
    Assistant: Let me help you with that. 
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return conversation_chain

def main():
    st.title("🍳 LangChain Recipe Assistant")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize components
    try:
        vectorstore = get_vectorstore()
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(vectorstore)
        
        # Chat interface
        st.write("👋 Hello! I'm your AI cooking assistant. I can help you find recipes, provide cooking tips, and answer your culinary questions!")
        
        # User input
        user_input = st.text_input("Ask me anything about cooking or recipes!", key="user_input")
        
        if user_input:
            # Get response from conversation chain
            response = st.session_state.conversation({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, response["answer"]))
            
            # Display chat history
            st.write("---")
            st.write("Chat History:")
            for user_msg, ai_msg in st.session_state.chat_history:
                st.write("You: " + user_msg)
                st.write("Assistant: " + ai_msg)
                st.write("---")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required API keys are set in the .env file and the vectorstore has been generated.")

if __name__ == "__main__":
    main()
