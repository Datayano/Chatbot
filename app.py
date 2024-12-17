import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
import json
import uuid

# Initialize SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    return QdrantClient(":memory:")  # Using in-memory storage for demonstration

# Function to load and process recipes
@st.cache_data
def load_recipes():
    try:
        df = pd.read_csv('sample_recipes.csv')
        return df
    except FileNotFoundError:
        st.error("sample_recipes.csv file not found! Please make sure it exists in the project directory.")
        return None

def setup_qdrant(client, model, recipes_df):
    collection_name = "recipes"
    
    # Create collection if it doesn't exist
    collections = client.get_collections().collections
    if not any(collection.name == collection_name for collection in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )
        
        # Generate embeddings and upload to Qdrant
        points = []
        for idx, row in recipes_df.iterrows():
            # Create a searchable text combining recipe details
            recipe_text = f"{row['name']} {row['Description']} {row['ingredients_name']}"
            embedding = model.encode(recipe_text)
            
            # Create a point using PointStruct with UUID
            payload = {k: str(v) if pd.isna(v) else v for k, v in row.to_dict().items()}
            payload['original_index'] = idx  # Store original index in payload
            point = PointStruct(
                id=str(uuid.uuid4()),  # Generate a UUID for each point
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Batch upsert all points
        client.upsert(
            collection_name=collection_name,
            points=points
        )

def format_recipe_details(recipe):
    """Format recipe details for display"""
    details = f"🍽️ **{recipe['name']}**\n\n"
    details += f"⏱️ Cooking time: {recipe['Cooking time']}\n"
    details += f"👥 Serves: {recipe['Covers count']} people\n"
    details += f"📝 Description: {recipe['Description']}\n\n"
    details += f"🥘 Ingredients:\n{recipe['ingredients_name']}\n\n"
    if 'URL' in recipe and recipe['URL']:
        details += f"🔗 [View full recipe]({recipe['URL']})\n"
    return details

def get_recipe_recommendations(query, client, model, top_k=3):
    query_vector = model.encode(query)
    
    # Search in Qdrant
    search_result = client.search(
        collection_name="recipes",
        query_vector=query_vector,
        limit=top_k
    )
    
    return search_result

def generate_response(user_input, context):
    # Simple rule-based response generation
    if "hello" in user_input.lower() or "hi" in user_input.lower():
        return "Hello! I'm your friendly recipe advisor. What kind of dish are you in the mood for today? I can suggest recipes based on cooking time, ingredients, or type of cuisine!"
    
    if "thank" in user_input.lower():
        return "You're welcome! Let me know if you need any other recipe suggestions!"
    
    if not context.get('recommendations'):
        return "I don't have any specific recommendations yet. Could you tell me what kind of dish you're looking for? You can specify cooking time, number of servings, or type of cuisine!"
    
    response = "Based on your request, here are some recipes I think you might enjoy:\n\n"
    for idx, rec in enumerate(context['recommendations'], 1):
        recipe = rec.payload
        response += format_recipe_details(recipe) + "\n"
        if idx < len(context['recommendations']):
            response += "---\n"
    
    response += "\nWould you like more details about any of these recipes? Or shall we look for something different? You can also ask about cooking time or serving sizes!"
    return response

# Main Streamlit app
def main():
    st.title("🍳 Your Friendly Recipe Chatbot")
    st.write("Hello! I'm your virtual recipe advisor. Ask me about recipes and I'll help you find the perfect dish!")

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.context = {}

    # Load components
    model = load_model()
    client = init_qdrant()
    recipes_df = load_recipes()

    if recipes_df is not None:
        # Setup Qdrant with recipes
        setup_qdrant(client, model, recipes_df)

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What kind of recipe are you looking for?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get recipe recommendations
            recommendations = get_recipe_recommendations(prompt, client, model)
            st.session_state.context['recommendations'] = recommendations

            # Generate and display assistant response
            assistant_response = generate_response(prompt, st.session_state.context)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)

if __name__ == "__main__":
    main()
