# Recipe Recommendation Chatbot

A friendly conversational chatbot that recommends recipes based on user preferences using Streamlit and Qdrant vector database.

## Features

- Interactive chat interface
- Recipe recommendations based on similarity search
- Natural conversation flow
- Vector-based recipe search using Qdrant
- Efficient embedding generation using SentenceTransformer

## Setup

1. Make sure you have Python installed and your virtual environment activated

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Prepare your recipes CSV file:
Create a file named `recipes.csv` with the following columns:
- name: Recipe name
- ingredients: List of ingredients
- instructions: Cooking instructions

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## How it Works

1. The app loads your recipes from the CSV file
2. Recipe details are converted into embeddings using SentenceTransformer
3. Embeddings are stored in Qdrant vector database
4. When you chat with the bot, it:
   - Converts your query to an embedding
   - Finds similar recipes using cosine similarity
   - Generates friendly responses with recommendations

## Note

The current implementation uses in-memory storage for Qdrant. For production use, you may want to configure a persistent Qdrant storage.
