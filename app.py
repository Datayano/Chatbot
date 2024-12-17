import streamlit as st
import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
import json
import uuid
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure OpenAI client for Grok API
openai.api_key = os.getenv('GROK_API_KEY')
openai.api_base = "https://api.groq.com/openai/v1"  # Grok API endpoint

# Initialize SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
@st.cache_resource
def init_qdrant():
    return QdrantClient(":memory:")

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
    
    collections = client.get_collections().collections
    if not any(collection.name == collection_name for collection in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )
        
        points = []
        for idx, row in recipes_df.iterrows():
            recipe_text = f"{row['name']} {row['Description']} {row['ingredients_name']}"
            embedding = model.encode(recipe_text)
            
            payload = {k: str(v) if pd.isna(v) else v for k, v in row.to_dict().items()}
            payload['original_index'] = idx
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
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
    
    search_result = client.search(
        collection_name="recipes",
        query_vector=query_vector,
        limit=top_k
    )
    
    return search_result

def generate_grok_response(user_input, context, recipes_df):
    # Create a system prompt that includes context about being a cooking assistant
    system_prompt = """You are a knowledgeable and friendly cooking assistant. You help users find recipes and provide cooking advice. 
    You have access to a recipe database and can make personalized recommendations. Be conversational and engaging, while providing 
    accurate and helpful cooking information. If you recommend recipes, explain why they might be good choices based on the user's preferences.
    You can also provide cooking tips, ingredient substitutions, and answer questions about cooking techniques."""

    # Create a context message that includes any recipe recommendations
    context_message = ""
    if context.get('recommendations'):
        context_message = "Based on your request, here are some recipes I found:\n"
        for rec in context['recommendations']:
            recipe = rec.payload
            context_message += f"- {recipe['name']}: {recipe['Description']}\n"

    # Combine user input with context for a complete conversation
    full_prompt = f"{context_message}\nUser: {user_input}\nAssistant:"

    try:
        # Get response from Grok using OpenAI-compatible API
        response = openai.ChatCompletion.create(
            model="mixtral-8x7b-32768",  # Grok's model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with Grok API: {str(e)}")
        return "I apologize, but I'm having trouble generating a response at the moment. Let me provide you with the recipe information I found."

def main():
    st.title("🍳 Intelligent Recipe Assistant")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Load components
    model = load_model()
    client = init_qdrant()
    recipes_df = load_recipes()
    
    if recipes_df is not None:
        setup_qdrant(client, model, recipes_df)
        
        # Chat interface
        st.write("👋 Hello! I'm your cooking assistant. I can help you find recipes, provide cooking tips, and answer your culinary questions!")
        
        # User input
        user_input = st.text_input("Ask me anything about cooking or recipes!", key="user_input")
        
        if user_input:
            # Get recipe recommendations
            recommendations = get_recipe_recommendations(user_input, client, model)
            
            # Create context for the conversation
            context = {
                'recommendations': recommendations,
                'chat_history': st.session_state.chat_history
            }
            
            # Generate response using Grok
            response = generate_grok_response(user_input, context, recipes_df)
            
            # Add to chat history
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response))
            
            # Display recipe details if available
            if recommendations:
                st.write("Here are some recipes that match your request:")
                for hit in recommendations:
                    recipe = hit.payload
                    with st.expander(f"📖 {recipe['name']}"):
                        st.markdown(format_recipe_details(recipe))
        
        # Display chat history
        st.write("---")
        st.write("Chat History:")
        for role, message in st.session_state.chat_history:
            if role == "user":
                st.write("You: " + message)
            else:
                st.write("Assistant: " + message)

if __name__ == "__main__":
    main()
