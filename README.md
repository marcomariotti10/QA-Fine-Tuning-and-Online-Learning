# 🧠 Natural Language Processing Project 2024: Question Answering on BeerQA

## 📋 Project Overview

This project explores the **BeerQA dataset**, focused on question answering (QA) tasks using documents from Wikipedia. The work includes:

- Dataset exploration and visualization
- Indexing and document retrieval using PyTerrier
- Word embeddings and clustering analysis
- Fine-tuning QA models like **FLAN-T5**, **BART**, and **Mini-LM**
- Evaluation using both ground truth contexts and retrieval pipelines
- Online learning simulation based on feedback-driven model updates

---

## 📚 Dataset

The project uses three JSON files from the BeerQA benchmark:

- `beerqa_train_v1.0.json`: Questions, contexts, and answers (train)
- `beerqa_dev_v1.0.json`: Development set
- `beerqa_test_questions_v1.0.json`: Unlabeled test questions

Each document contains:
- A question
- A list of title-paragraph context pairs
- An answer (when labeled)

---

## 🔍 Components

### 📊 Dataset Exploration
- Analyze question and context length
- Count documents by number of contexts
- Visualize statistics and answer frequency
- Identify paragraph-title reuse

### 📁 Indexing with PyTerrier
- Index contexts using BM25, TF-IDF, PL2
- Rank paragraphs per query using multi-stage retrieval
- Evaluate retrieval visually using bubble plots and rank comparisons

### 📐 Embeddings and Clustering
- Train Word2Vec embeddings on questions, contexts, and answers
- Use t-SNE to visualize embedding space
- Perform document clustering with TF-IDF + KMeans
- Assess cluster quality by title and content consistency

### 🤖 Question Answering Models
- Fine-tune **FLAN-T5-small** and **BART-base**
- Evaluate with and without title context
- Compare multiple pretrained QA pipelines (e.g., RoBERTa, ELECTRA, Mini-LM)
- Metrics: Exact Match, F1 Score

### 🔁 Online Learning Simulation
- Simulate user feedback using unseen residual data
- Incrementally fine-tune **Mini-LM** on mistakes
- Track performance over iterations (F1 and EM)
- Visualize progressive model improvement

---

## 🧪 Evaluation Metrics

Implemented standard QA metrics:
- **Exact Match (EM)**: Checks if prediction exactly matches the reference
- **F1 Score**: Measures overlap between predicted and actual tokens
- **Rouge-L**: Used for generative models like T5 and BART

---

## 📦 Requirements

To run this project, you'll need:

- Python 3.7+
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Datasets (🤗)
- TensorFlow
- PyTerrier
- SentenceTransformers
- scikit-learn
- Matplotlib / Seaborn / Plotly
- gensim
- hnswlib
- pandas / NumPy
