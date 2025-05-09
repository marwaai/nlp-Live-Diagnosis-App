🧠 Live Diagnosis App
An AI-powered medical Q&A assistant that uses local documents and a lightweight local language model to deliver real-time, streaming answers to health-related questions. Built with LangChain, Streamlit, FAISS, and CTransformers, with optional backend deployment using Flask, Socket.IO, and Gunicorn.

💡 Features
🔎 Context-Aware QA: Uses Retrieval-Augmented Generation (RAG) with FAISS and HuggingFace embeddings.

📂 Custom Knowledge Base: Load your own .txt documents for domain-specific responses.

⏱️ Live Token Streaming: Responses update in real-time using Streamlit and custom callback handlers.

🧠 Local LLM Inference: No API key required — powered by CTransformers and GGML models.

🌐 Optional Flask Deployment: Easily extend or deploy via Flask + Gunicorn + WebSockets.

📁 Project Structure
graphql
Copy
Edit
.
├── db/                      # Directory containing .txt files and FAISS index
│   ├── your_files.txt
│   └── faiss_index.pkl
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── model/                  # Directory to store your LLM file
    └── hermeslimarp-l2-7b.ggmlv3.q2_K.bin
🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/live-diagnosis-app.git
cd live-diagnosis-app
2. Set Up Your Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add Your Data
Put your .txt medical documents inside the db/ folder. These will be used to generate the FAISS index.

5. Download the Model
Download hermeslimarp-l2-7b.ggmlv3.q2_K.bin from TheBloke on Hugging Face and place it in the model/ directory (or wherever you configure CTransformers to load from).

▶️ Run the App (Streamlit)
bash
Copy
Edit
streamlit run app.py
Then open your browser at http://localhost:8501.

🧪 Requirements
Your requirements.txt:

txt
Copy
Edit
flask_socketio==5.3.6
langchain==0.2.3
langchain_community==0.2.4
sentence-transformers==3.0.1
ctransformers==0.2.27
faiss-cpu==1.8.0
Flask==2.2.5
gunicorn==22.0.0
gevent==24.2.1
gevent-websocket==0.10.1
streamlit==1.35.0
⚠️ Note: If you’re using extra LangChain integrations (like HuggingFace), ensure they’re installed via the correct plugin packages or langchain_community.

🌐 Optional Flask + Gunicorn Deployment
You can adapt the app for Flask + Socket.IO if needed for production-grade deployment. A minimal gunicorn startup command would be:

bash
Copy
Edit
gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app
Make sure your app.py is structured as a Flask app if using that route (not just Streamlit).

📜 License
MIT License — you’re free to use, modify, and distribute this app with attribution.
