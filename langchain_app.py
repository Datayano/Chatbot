"""
Application d'assistant culinaire utilisant LangChain et RAG (Retrieval Augmented Generation)

Ce script implémente un assistant culinaire intelligent qui utilise:
- LangChain: Framework pour développer des applications basées sur les LLM (Large Language Models)
- RAG: Technique permettant d'enrichir les réponses du LLM avec des données externes
- Streamlit: Bibliothèque pour créer des interfaces web interactives en Python

L'assistant peut:
- Recommander des recettes basées sur les ingrédients ou préférences
- Répondre aux questions culinaires en utilisant une base de connaissances
- Formater les réponses en markdown pour une meilleure lisibilité
"""

# Import des bibliothèques nécessaires
from langchain_community.vectorstores import Chroma  # Pour la base de données vectorielle
from langchain_openai import OpenAIEmbeddings     # Pour convertir le texte en vecteurs
from langchain_groq import ChatGroq              # LLM de Groq (alternative à GPT)
from langchain.memory import ConversationBufferMemory  # Pour gérer l'historique des conversations
from langchain.chains import ConversationalRetrievalChain  # Pour combiner recherche et conversation
from langchain.prompts import PromptTemplate    # Pour structurer les prompts
from dotenv import load_dotenv  # Pour gérer les variables d'environnement
import streamlit as st  # Pour l'interface utilisateur
import os

# Chargement des variables d'environnement (clés API, etc.)
load_dotenv()

# Initialisation du modèle d'embedding
@st.cache_resource  # Cache la ressource pour éviter de la recharger à chaque requête
def get_embeddings():
    """
    Initialise le modèle d'embedding d'OpenAI.
    
    Les embeddings sont des représentations vectorielles du texte qui permettent
    de mesurer la similarité sémantique entre différents textes.
    
    Returns:
        OpenAIEmbeddings: Instance du modèle d'embedding configuré
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY non trouvée dans les variables d'environnement")
    return OpenAIEmbeddings(
        model="text-embedding-3-small"  # Modèle plus léger et économique
    )

# Initialisation du modèle de langage (LLM)
@st.cache_resource
def get_llm():
    """
    Initialise le modèle de langage Groq.
    
    Groq est une alternative à GPT d'OpenAI, offrant des performances similaires
    avec potentiellement des coûts différents.
    
    Returns:
        ChatGroq: Instance du modèle de langage configuré
    """
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY non trouvée dans les variables d'environnement")
    return ChatGroq(
        temperature=0.7,  # Contrôle la créativité des réponses (0=conservateur, 1=créatif)
        model_name="mixtral-8x7b-32768",  # Modèle Mixtral, un des plus performants de Groq
        max_tokens=32768  # Longueur maximale des réponses
    )

# Chargement de la base de données vectorielle
@st.cache_resource
def get_vectorstore():
    """
    Charge la base de données vectorielle Chroma.
    
    Chroma est une base de données vectorielle qui stocke les embeddings des documents
    et permet de faire des recherches par similarité sémantique.
    
    Returns:
        Chroma: Instance de la base de données vectorielle
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory="chroma_db",  # Dossier où sont stockés les vecteurs
        embedding_function=embeddings
    )
    return vectorstore

# Création de la chaîne de conversation
def get_conversation_chain(vectorstore):
    """
    Configure la chaîne de conversation qui combine recherche et dialogue.
    
    Cette fonction:
    1. Initialise la mémoire pour garder le contexte de la conversation
    2. Crée un template de prompt qui guide le comportement de l'assistant
    3. Configure la chaîne de conversation qui utilise le LLM et la recherche
    
    Args:
        vectorstore: Base de données vectorielle pour la recherche
        
    Returns:
        ConversationalRetrievalChain: Chaîne de conversation configurée
    """
    # Initialisation de la mémoire pour le contexte
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Création du template de prompt
    template = """Tu es un assistant culinaire sympathique et compétent. Utilise les éléments de contexte suivants pour 
    fournir des recommandations de recettes et des conseils de cuisine utiles. 
    
    Pour chaque recette, utilise ce format markdown:
    ### 🍽️ [Nom de la recette]
    **⏱️ Temps de cuisson:** [temps]
    **📊 Difficulté:** [niveau]
    
    #### 🥘 Ingrédients
    - [ingrédient 1]
    - [ingrédient 2]
    ...
    
    #### 📝 Instructions
    [instructions détaillées]
    
    ---
    
    Pour les questions générales sur la cuisine, utilise du markdown avec des titres (##, ###), 
    des listes (- ou *), et du texte en gras (**) ou en italique (*) quand c'est approprié.
    Réponds toujours en français.

    <contexte> 
    {context}
    </contexte>
    
    Historique de conversation: {chat_history}
    
    Humain: {question}
    
    Assistant: Je vais t'aider avec ça."""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Configuration de la chaîne de conversation
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Récupère les 3 documents les plus pertinents
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return conversation_chain

def main():
    """
    Fonction principale qui configure et lance l'interface utilisateur Streamlit.
    
    Cette fonction:
    1. Configure la page Streamlit
    2. Initialise les composants nécessaires (vectorstore, conversation)
    3. Gère l'interface utilisateur et les interactions
    """
    # Configuration de la page Streamlit
    st.set_page_config(
        page_title="Assistant Culinaire",
        page_icon="🍳",
        layout="wide"
    )
    
    st.title("🧑‍🍳 Assistant Culinaire LangChain")
    
    # Styles CSS pour l'interface
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
    .assistant-message {
        background-color: #f5f5f5;
        margin-left: 5%;
        margin-right: 20%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialisation des variables de session
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialisation des composants
    try:
        # Chargement de la base de données vectorielle
        vectorstore = get_vectorstore()
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(vectorstore)
        
        # Interface de chat
        st.markdown("""
        ### 👋 Bienvenue sur votre Assistant Culinaire IA !
        Je peux vous aider à trouver des recettes, donner des conseils de cuisine et répondre à vos questions culinaires.
        Essayez de me demander par exemple :
        - "Je veux cuisiner quelque chose avec du poulet et des légumes"
        - "Quelle est une recette rapide de pâtes ?"
        - "Donne-moi une recette pour un dîner végétarien"
        """)
        
        # Champ de saisie utilisateur
        user_input = st.text_input(
            "Que souhaitez-vous cuisiner aujourd'hui ?",
            key="user_input", 
            placeholder="Posez-moi n'importe quelle question sur la cuisine ou les recettes !"
        )
        
        if user_input:
            # Traitement de la requête utilisateur
            with st.spinner('Recherche de la recette parfaite...'):
                response = st.session_state.conversation({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })
                
                # Mise à jour de l'historique
                st.session_state.chat_history.append((user_input, response["answer"]))

            # Affichage de la réponse avec formatage markdown
            st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
            st.markdown(response["answer"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Affichage de l'historique des conversations dans un expander
            with st.expander("💬 Historique des conversations", expanded=False):
                st.markdown("""
                <style>
                .chat-message-user {
                    background-color: #e3f2fd;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 20% 5px 0;
                }
                .chat-message-assistant {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px 0 5px 20%;
                }
                .timestamp {
                    font-size: 0.8em;
                    color: #666;
                    margin-bottom: 5px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                for i, (user_msg, ai_msg) in enumerate(reversed(st.session_state.chat_history)):
                    # Message utilisateur
                    st.markdown(f'<div class="chat-message-user">', unsafe_allow_html=True)
                    st.markdown(f"**👤 Vous:** {user_msg}", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Message assistant
                    st.markdown(f'<div class="chat-message-assistant">', unsafe_allow_html=True)
                    st.markdown(f"**🤖 Assistant:** {ai_msg}", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Séparateur entre les conversations
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
            
    except Exception as e:
        # Gestion des erreurs
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
