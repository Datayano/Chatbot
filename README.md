# Application d'assistant culinaire

Ce projet permet d'intéragir avec un assistant culinaire amusant et convivial. Il retient l'historique des conversations et est capable de recommander des recettes provenant d'un fichier CSV converti en base de données vectorielle, en fonction des demandes et préférances de l'utilisateur.

## Fonctionnalités

- Assistant culinaire sous forme de chat.
- Recommandations de recettes basées sur la similarité grace a une base de données vectorielle et au cosine similarity.
- Conversation naturelle entre l'utilisateur et l'assistant, car l'assistant est géré par un LLM (Large Language Model).
- Les embeddings qui servent à faire des recherches par similarité dans la base de données vectorielle sont faits avec OpenAI qui est rapide, et très efficace. A noter qu'il existe des alternatives gratuites (beaucoup plus lentes et moins performantes à l'heure actuelle).

## Setup

### 1. Clonez le repository:
```bash
git clone https://github.com/Datayano/Chatbot.git
cd Chatbot
```
Ceci créera un dossier `Chatbot` dans lequel vous trouverez tous les fichiers du projet.

### 2. Creez votre environnement virtuel et activez-le avec ces 2 commandes:
Sous windows depuis un terminal Powershell ou celui de VSCode :
```bash
python -m venv chatbot_env
chatbot_env/Scripts/activate.ps1
```
Sous linux ou mac :
```bash
python3 -m venv chatbot_env
source chatbot_env/bin/activate
```

Sur VSCode, CTR+Shift+P pour ouvrir le panneau de commandes puis selectionez "Python: Select Interpreter" et choisissez l'environnement virtuel que vous venez de créer.

### 3. Installez les packages necessaires:
Sous Windows :
```bash
pip install -r requirements.txt
```
Sous Mac ou Linux :
```bash
pip3 install -r requirements.txt
```

### 4. Configurez les variables d'environnement:
Renommez le fichier sample.env en .env et remplissez les 2 variables d'environnement avec vos clés API. (OpenAI et XAI)

### 5. Lancez votre application Streamlit:
```bash
streamlit run langchain_app.py
```

## Comment ça marche ?

1. La première fois que vous lancez l'application, elle detecte que la base de données vectorielle n'existe pas encore et elle va la créer. Ensuite, elle va charger les recettes depuis le CSV, puis les transformer en une liste documents, et enfin transformer chaque document (ou recette) en vecteurs (embeddings) pour les stocker dans la base de données vectorielle ChromaDB nouvellement crée. Cela peut prendre quelques minutes mais il n'y aura plus besoin de le refaire après.
2. Ensuite, la page Streamlit s'ouvre et vous pouvez utiliser l'assistant culinaire pour trouver des recettes ou des conseils culinaires.
3. Lorsque vous conversez avec l'assistant culinaire, l'application va :
- transformer votre question en embeddings, 
- trouver les recettes similaires dans la base de données vectorielle 
- et vous donner des recommandations personnalisées basées sur sa recherche dans la base de données vectorielle.

## Version Jupyter Notebook pour bien comprendre !

Le notebook `langchain_rag_tutorial.ipynb` est une version du projet, sans streamlit, et sans interface utilisateur. 
C'est une version simplifiée et épurée qui vous permettra de comprendre le fonctionnement de l'assistant culinaire. 
Ce notebook est indépendant de l'application Streamlit mais il en reprend fidèlement toute la trame. 
Il utilise sa propre base de données vectorielle, comme ça si vous faites des testes sur le notebook, vous ne risquez pas de corrompre la base de donnée de votre streamlit. 
Ce notebook utilise aussi le LLM Grok, et l'embedder d'OpenAI. Il est donc parfait pour vous aider à comprendre le fonctionnement de l'assistant culinaire. 
Pour fonctionner, lui aussi aura besoin des clés API OpenAI et XAI présentes dans le fichier .env.


## Outils utilisés

Langchain : Framework pour developper des applications basé sur les LLM
ChromaDB : Base de données vectorielle pour stocker et rechercher des documents
OpenAI : Embeddings pour transformer le texte en vecteurs
Streamlit : Bibliothèque pour créer des interfaces web interactives en Python
Grok : LLM de xAI
Pydantic : Framework de validation de données
Pandas : Manipulation de données

## Structure du projet

### Principaux fichiers :
- `generate_vectorstore.py` : Script de génération de la base de données vectorielle. (géré automatiquement)
- `langchain_app.py` : Application Streamlit pour l'assistant culinaire.
- `langchain_rag_tutorial.ipynb` : Notebook Jupyter pour comprendre le fonctionnement de l'assistant culinaire.

### Arbre de fichiers :
```	
📂Chatbot/                      # Dossier principal du projet
📂├── .chatbot_env/             # Dossier contenant votre environnement virtuel
📂├── chroma_db/                # Dossier contenant la base de données vectorielle utilisée par Streamlit
📂├── chroma_db_jupiternotebook/ # Dossier contenant la base de données vectorielle utilisée par le notebook
📝├── generate_vectorstore.py   # Script de génération de la base de données vectorielle
🚀├── langchain_app.py          # Application Streamlit pour l'assistant culinaire
📝├── langchain_rag_tutorial.ipynb # Notebook Jupyter pour comprendre le fonctionnement de l'assistant culinaire
📝├── sample_recipes.csv        # Fichier CSV contenant les recettes
🔒├── .env                      # Fichier d'environnement contenant vos 2 clés API
📋├── requirements.txt          # Fichier contenant les packages nécessaires
🎨├── style.css                 # Fichier CSS pour la mise en forme de l'interface utilisateur
📝├── .gitignore                # Fichier pour ignorer certains fichiers et dossiers
📜└── README.md                 # Fichier de documentation du projet
```

## Créez vos clés API OpenAI et XAI

Pour utiliser l'assistant culinaire, vous devez avoir des clés API OpenAI et XAI. 
Pour les générer :
- https://console.x.ai/
- https://platform.openai.com/signup
