"""
Script de génération de la base de données vectorielle pour l'assistant culinaire

Ce script transforme un fichier CSV contenant des recettes en une base de données vectorielle
utilisable par notre assistant culinaire. Il utilise plusieurs concepts clés :

1. Embeddings : Conversion de texte en vecteurs numériques permettant de mesurer
   la similarité sémantique entre différents textes.
   
2. Vectorstore : Base de données spécialisée qui stocke ces vecteurs et permet
   de faire des recherches par similarité.
   
3. RAG (Retrieval Augmented Generation) : Technique qui permet d'enrichir les réponses
   d'un LLM avec des données externes (ici, notre base de recettes).
"""

import pandas as pd  # Pour la manipulation des données
from langchain_community.vectorstores import Chroma  # Base de données vectorielle
from langchain_openai import OpenAIEmbeddings  # Pour créer les embeddings
from langchain.schema import Document  # Structure de données pour les documents
from typing import List  # Pour le typage des fonctions
import os
from dotenv import load_dotenv  # Pour gérer les variables d'environnement
import shutil  # Pour les opérations sur les fichiers

# Chargement des variables d'environnement (clés API)
load_dotenv()

def create_documents_from_csv(csv_path: str) -> List[Document]:
    """
    Crée une liste de documents à partir d'un fichier CSV de recettes.
    
    Cette fonction:
    1. Lit le fichier CSV contenant les recettes
    2. Pour chaque recette, crée un document structuré avec:
       - Le contenu (texte de la recette)
       - Les métadonnées (temps de cuisson, nombre de personnes, etc.)
    
    Args:
        csv_path (str): Chemin vers le fichier CSV des recettes
        
    Returns:
        List[Document]: Liste des documents structurés prêts à être vectorisés
    """
    # Lecture du fichier CSV
    df = pd.read_csv(csv_path)
    documents = []
    
    # Traitement de chaque ligne du CSV
    for _, row in df.iterrows():
        # Combinaison des champs textuels pour créer le contenu
        content = f"Recipe: {row['name']}\n\nDescription: {row['Description']}\n\nIngredients: {row['ingredients_name']}"
        
        # Création des métadonnées associées
        metadata = {
            'cooking_time': row['Cooking time'],
            'covers_count': row['Covers count'],
            'url': row['URL'] if 'URL' in row else '',
            'source': 'recipe_database'
        }
        
        # Création du document structuré
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def main():
    """
    Fonction principale qui gère la création de la base de données vectorielle.
    
    Cette fonction:
    1. Vérifie la présence de la clé API OpenAI
    2. Initialise le modèle d'embedding
    3. Crée les documents à partir du CSV
    4. Génère et sauvegarde la base de données vectorielle
    """
    # Vérification de la clé API
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Clé API OpenAI non trouvée dans les variables d'environnement")
    
    # Possibilité de nettoyer la base existante (actuellement commenté)
    #if os.path.exists("./chroma_db"):
    #    shutil.rmtree("./chroma_db")
    
    # Initialisation du modèle d'embedding
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Utilisation du modèle plus léger et économique
    )
    
    # Création des documents à partir du CSV
    documents = create_documents_from_csv("sample_recipes.csv")
    
    # Création de la base de données vectorielle
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Dossier où sera sauvegardée la base
    )
    
    # Sauvegarde permanente de la base
    vectorstore.persist()
    print(f"Base de données vectorielle créée avec {len(documents)} documents et sauvegardée dans ./chroma_db")

if __name__ == "__main__":
    main()
