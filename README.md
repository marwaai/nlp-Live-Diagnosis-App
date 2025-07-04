
# 🩺 Live Diagnosis App

**Live Diagnosis App** is a real-time, locally-run medical chatbot built using **LangChain**, **FAISS**, **CTransformers**, and **Streamlit**. It allows users to ask medical questions and receive live-streamed answers generated from a local LLM and your custom document database.

---

## 🚀 Features

* 🧠 Locally hosted LLM (HermesLimaRP L2 7B) using GGML + `CTransformers`
* 🔍 RAG (Retrieval-Augmented Generation) with FAISS vector store
* 📝 Real-time streaming of answers in Streamlit using a custom callback handler
* 📂 Supports dynamic `.txt` document ingestion and chunking
* 🗂 Embeddings generated via HuggingFace (`all-MiniLM-L6-v2`)
* 🔄 Automatically caches and reuses vector store (FAISS)

---

## 🧰 Requirements

Install dependencies:

```bash
pip install streamlit langchain faiss-cpu ctransformers sentence-transformers
```

You also need:

* Local LLM model: `hermeslimarp-l2-7b.ggmlv3.q2_K.bin`
* Model should be located in the same directory or update the path
* A folder called `db/` containing your `.txt` medical documents

---

## 🗃️ Project Structure

```
.
├── app.py                          # Main Streamlit script
├── db/
│   ├── custom_doc1.txt
│   ├── custom_doc2.txt
│   └── faiss_index.pkl             # Serialized FAISS index (auto-generated)
├── hermeslimarp-l2-7b.ggmlv3.q2_K.bin  # Local GGML quantized model
└── README.md
```

---

## ⚙️ How It Works

1. Loads all `.txt` files in `db/`
2. Splits them into chunks using `RecursiveCharacterTextSplitter`
3. Generates vector embeddings using HuggingFace
4. Stores/loads vectors from `faiss_index.pkl`
5. Queries are passed to the local LLM (HermesLimaRP) with retrieval support
6. Answers are streamed live in Streamlit UI

---

## 🖥️ How to Run

1. Make sure your model and document files are in place.
2. Run the app:

```bash
streamlit run app.py
```

3. Open the app in your browser (usually [http://localhost:8501](http://localhost:8501))

---

## 🧪 Example Usage

**User input:**

```text
What are common symptoms of vitamin B12 deficiency?
```

**Live answer (streamed):**

```text
Vitamin B12 deficiency can cause fatigue, weakness, memory issues, numbness or tingling in hands and feet, and mood changes. In severe cases, it may lead to anemia or neurological problems.
```

---

## 💡 Customization Tips

* Replace or add `.txt` files in `db/` to change the knowledge base.
* Increase `search_kwargs={"k": 1}` to 3–5 for broader retrieval.
* Swap model file with any GGML-compatible quantized LLM (e.g., Mistral, LLaMA2).
* Modify prompt for different assistant personalities (doctor, teacher, etc.)

---

## ⚠️ Disclaimer

This is a **prototype** for educational purposes. It does **not** replace professional medical advice. Use responsibly.



