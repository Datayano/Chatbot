import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List
import os
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

def create_documents_from_csv(csv_path: str) -> List[Document]:
    """
    Create documents from CSV with content and metadata separation
    """
    df = pd.read_csv(csv_path)
    documents = []
    
    for _, row in df.iterrows():
        # Combine content fields
        content = f"Recipe: {row['name']}\n\nDescription: {row['Description']}\n\nIngredients: {row['ingredients_name']}"
        
        # Create metadata
        metadata = {
            'cooking_time': row['Cooking time'],
            'covers_count': row['Covers count'],
            'url': row['URL'] if 'URL' in row else '',
            'source': 'recipe_database'
        }
        
        # Create Document object
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Clean up existing vectorstore if it exists
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    # Initialize embedding model with a more cost-effective model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Using the smaller, more cost-effective model
    )
    
    # Create documents
    documents = create_documents_from_csv("sample_recipes.csv")
    
    # Create and persist vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Persist the vectorstore
    vectorstore.persist()
    print(f"Vectorstore created with {len(documents)} documents and persisted to ./chroma_db")

if __name__ == "__main__":
    main()
