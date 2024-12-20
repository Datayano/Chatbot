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
    recipe_name: str = Field(description="Nom de la recette")
    cooking_time: str = Field(description="Temps de cuisson estimé")
    ingredients: List[str] = Field(description="Liste des ingrédients principaux")
    instructions: Optional[str] = Field(description="Instructions de cuisson brèves")
    difficulty: str = Field(description="Niveau de difficulté (Facile, Moyen, Difficile)")
    
    class Config:
        schema_extra = {
            "example": {
                "recipe_name": "Spaghetti Carbonara",
                "cooking_time": "20 minutes",
                "ingredients": ["spaghetti", "œufs", "fromage pecorino", "guanciale", "poivre noir"],
                "instructions": "Cuire les pâtes, préparer la sauce avec les œufs et le fromage, mélanger avec le guanciale croustillant",
                "difficulty": "Moyen"
            }
        }

class RecipeList(BaseModel):
    recipes: List[RecipeRecommendation] = Field(description="Liste des recettes")
    
    class Config:
        schema_extra = {
            "example": {
                "recipes": [
                    {
                        "recipe_name": "Spaghetti Carbonara",
                        "cooking_time": "20 minutes",
                        "ingredients": ["spaghetti", "œufs", "fromage pecorino", "guanciale", "poivre noir"],
                        "instructions": "Cuire les pâtes, préparer la sauce avec les œufs et le fromage, mélanger avec le guanciale croustillant",
                        "difficulty": "Moyen"
                    },
                    {
                        "recipe_name": "Risotto aux Champignons",
                        "cooking_time": "30 minutes",
                        "ingredients": ["riz arborio", "champignons", "oignon", "vin blanc", "bouillon de légumes"],
                        "instructions": "Faire revenir l'oignon, ajouter le riz, déglacer au vin blanc, ajouter le bouillon petit à petit",
                        "difficulty": "Moyen"
                    }
                ]
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
    parser = PydanticOutputParser(pydantic_object=RecipeList)
    
    # Create custom prompt template
    template = """Tu es un assistant culinaire sympathique et compétent. Utilise les éléments de contexte suivants pour 
    fournir des recommandations de recettes et des conseils de cuisine utiles. 
    
    Si on te demande une ou plusieurs recettes, renvoie-les dans une liste formatée selon le schéma JSON spécifié.
    Si on te pose une question générale sur la cuisine, réponds en utilisant du markdown avec des titres (##, ###), 
    des listes (- ou *), et du texte en gras (**) ou en italique (*) quand c'est approprié.

    Contexte: {context}
    
    Historique de conversation: {chat_history}
    
    Humain: {question}
    
    Assistant: Je vais t'aider avec ça. 
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
    st.set_page_config(
        page_title="Assistant Culinaire",
        page_icon="🍳",
        layout="wide"
    )
    
    st.title("🧑‍🍳 Assistant Culinaire LangChain")
    st.markdown("""
    <style>
    .recipe-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f7ff;
        margin: 10px 0;
        border: 1px solid #cce0ff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ingredient-list {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
    }
    .recipe-header {
        color: #2c5282;
        font-size: 1.5em;
        margin-bottom: 15px;
    }
    .recipe-metadata {
        display: inline-block;
        margin: 5px 10px;
        padding: 5px 10px;
        background-color: #ffffff;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
    }
    .instructions-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin-top: 10px;
    }
    .chat-message {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
        margin-right: 5%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-left: 5%;
        margin-right: 20%;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
        st.markdown("""
        ### 👋 Bienvenue sur votre Assistant Culinaire IA !
        Je peux vous aider à trouver des recettes, donner des conseils de cuisine et répondre à vos questions culinaires.
        Essayez de me demander par exemple :
        - "Je veux cuisiner quelque chose avec du poulet et des légumes"
        - "Quelle est une recette rapide de pâtes ?"
        - "Donne-moi une recette pour un dîner végétarien"
        """)
        
        # User input
        user_input = st.text_input("Que souhaitez-vous cuisiner aujourd'hui ?", key="user_input", 
                                 placeholder="Posez-moi n'importe quelle question sur la cuisine ou les recettes !")
        
        if user_input:
            with st.spinner('Recherche de la recette parfaite...'):
                # Get response from conversation chain
                response = st.session_state.conversation({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })
                
                # Add to chat history
                st.session_state.chat_history.append((user_input, response["answer"]))
                st.write("hello")
                st.write(type(response["answer"]))
            # Display current conversation
            
            st.markdown("### 📝 Dernières Recommandations")
            
            try:
                # Try to parse the response as a RecipeList
                recipe_list = RecipeList.parse_raw(response["answer"])
                
                # Display each recipe in a nice card format
                for recipe in recipe_list.recipes:
                    with st.container():
                        st.markdown(f"""
                        <div class="recipe-card">
                            <div class="recipe-header">🍽️ {recipe.recipe_name}</div>
                            <div class="recipe-metadata">⏱️ {recipe.cooking_time}</div>
                            <div class="recipe-metadata">📊 {recipe.difficulty}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🥘 Ingrédients")
                            with st.container():
                                st.markdown('<div class="ingredient-list">', unsafe_allow_html=True)
                                for ingredient in recipe.ingredients:
                                    st.markdown(f"• {ingredient}")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 📝 Instructions")
                            if recipe.instructions:
                                st.markdown('<div class="instructions-box">', unsafe_allow_html=True)
                                st.markdown(recipe.instructions)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add some spacing between recipes
                        st.markdown("<br>", unsafe_allow_html=True)
            
            except Exception as e:
                # If parsing fails, format the response as markdown
                response_text = response["answer"]
                
                # Remove any JSON formatting artifacts if present
                response_text = response_text.replace('```json', '').replace('```', '')
                
                # Display the response in a styled container
                st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
                st.markdown(response_text)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display chat history in a collapsible section
            with st.expander("💬 Voir l'historique de conversation", expanded=False):
                for user_msg, ai_msg in st.session_state.chat_history:
                    st.markdown('<div class="chat-message user-message">', unsafe_allow_html=True)
                    st.markdown(f"**Vous:** {user_msg}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
                    st.markdown(f"**Assistant:** {ai_msg}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error("⚠️ Erreur de Configuration")
        st.error(f"Une erreur s'est produite : {str(e)}")
        st.markdown("""
        Veuillez vérifier :
        1. Que toutes les clés API requises sont définies dans le fichier `.env`
        2. Que le vectorstore a été généré en utilisant `generate_vectorstore.py`
        3. Que les dépendances nécessaires sont installées
        """)

if __name__ == "__main__":
    main()
