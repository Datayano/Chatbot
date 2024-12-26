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
from openai import OpenAI                        # Client OpenAI pour Grok
from langchain_openai import ChatOpenAI          # Pour utiliser Grok avec LangChain
from langchain.memory import ConversationBufferMemory  # Pour gérer l'historique des conversations
from langchain_core.prompts import PromptTemplate    # Pour structurer les prompts
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain  # Pour combiner recherche et conversation
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
    Initialise le modèle de langage Grok.
    
    Grok est un modèle de langage développé par xAI,
    offrant des performances de haut niveau pour la génération de texte.
    
    Returns:
        ChatOpenAI: Instance du modèle de langage configuré
    """
    if not os.getenv("XAI_API_KEY"):
        raise ValueError("XAI_API_KEY non trouvée dans les variables d'environnement")
    
    return ChatOpenAI(
        temperature=0.7,  # Contrôle la créativité des réponses (0=conservateur, 1=créatif)
        model_name="grok-2-1212",  # Modèle Grok
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
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
    
    Instructions importantes:
    - Ne jamais inclure d'URLs ou de liens vers des sites externes dans tes réponses
    - Réponds toujours en français
    - Utilise du markdown avec des titres (##, ###), des listes (- ou *), et du texte en gras (**) ou en italique (*) quand c'est approprié

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
    
    # Chargement du CSS depuis le fichier externe
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
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
                response = st.session_state.conversation.invoke({
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
