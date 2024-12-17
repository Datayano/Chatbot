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

2. Install rustup-init (needed by Qdrant)

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Prepare your recipes CSV file:
Create a file named `recipes.csv` with the following columns:
- name: Recipe name
- ingredients: List of ingredients
- instructions: Cooking instructions

5. Run the Streamlit app:
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


# Workflow

The project is managed by LangChain, which allows for the use of the following components:

Connection to a CSV file and transformation into a document format.
LLM with Groq.
Integration of the LLM with a JSON output parser, using Pydantic to define the output format.
Embeddings: Sentence Transformers, Cohere Embeddings, or OpenAI Embeddings.
Vectorstore: ChromaDB.
Retriever: To match user input with the vectorstore.
The retriever's output is processed by an LLM to generate a clean and structured result.
The entire workflow is managed by a chatbot with integrated memory.

# A modifier


text-embedding-3-small 
0,02 /1M

