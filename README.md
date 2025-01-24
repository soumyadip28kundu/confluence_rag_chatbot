# confluence_rag_chatbot
A Ollama based chatbot which takes data only from confluence pages and hosted locally. Hosted as a streamit app supporting rag based architecture.

## Prerequisites
1. Create a confluence account
2. Download Ollama engine for Local LLM setup. Ollama can be downloaded from [Ollama](https://ollama.com/)

## Steps to follow

### Confluence
Create authorization for your confluence account. Go to Profile-> Manage Account -> Security -> Create and managed API Token to create your API token

### Ollama
Once installed Ollama should be running in the background. 

Pull the models llama3.2:latest and nomic-embed-text:latest

ollama pull nomic-embed-text:latest

ollama pull llama3.2:latest

### Steps to run the project

1. Install the project requirements by running
   ```
   pip install -r requirements.txt
   ```
2. Open the config.ini and populate the confluence details
  ```
  [confluence_credentials]
  url=<<your_confluence page url>>
  username=<<username>>
  passkey=<<generated confluence pass key>>
  spacekey=<<confluence spaece key>>
```
3. Run the confluence_to_chromadb.py.

   This will create the loacal vector embedding based on nomic-embed-text.

4. Run the streamlit_rag_app.py

   streamlit run streamlit_rag_app.py
