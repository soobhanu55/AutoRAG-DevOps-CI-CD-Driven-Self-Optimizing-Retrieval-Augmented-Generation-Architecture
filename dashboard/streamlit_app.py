import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Auto-RAG-Devops Dashboard", layout="wide")

st.title("Auto-RAG-Devops Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Query", "Ingestion", "Optimization", "Metrics"])

if page == "Query":
    st.header("Test Retrieval Pipeline")
    
    # Get current config
    try:
        config_res = requests.get(f"{API_URL}/config")
        if config_res.status_code == 200:
            config = config_res.json()
            st.info(f"Active Retriever: **{config['active_retriever']}** | Reranker Enabled: **{config['reranker_enabled']}**")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Ensure it is running on port 8000.")
    
    query = st.text_input("Enter your question:")
    if st.button("Search"):
        if query:
            with st.spinner("Retrieving and generating answer..."):
                try:
                    res = requests.post(f"{API_URL}/query", json={"query": query, "top_k": 5})
                    if res.status_code == 200:
                        data = res.json()
                        
                        st.subheader("Answer")
                        st.write(data["answer"])
                        
                        st.subheader("Metrics")
                        metrics_df = pd.DataFrame([data["metrics"]])
                        st.dataframe(metrics_df)
                        
                        st.subheader("Retrieved Context")
                        for i, doc in enumerate(data["context"]):
                            with st.expander(f"Context {i+1} (Score: {doc.get('score', doc.get('rerank_score', 'N/A'))})"):
                                st.write(doc["text"])
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to fetch: {e}")

elif page == "Ingestion":
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md"])
    strategy = st.selectbox("Chunking Strategy", ["fixed", "semantic", "sliding_window"])
    
    if st.button("Ingest File") and uploaded_file:
        with st.spinner("Ingesting and processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{API_URL}/ingest", params={"strategy": strategy}, files=files)
                if res.status_code == 200:
                    st.success(f"Successfully ingested {uploaded_file.name}")
                    st.json(res.json())
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

elif page == "Optimization":
    st.header("Run Auto-Optimization")
    st.write("Trigger grid-search to find the best performing pipeline configuration using RAGAS.")
    
    q_input = st.text_area("Validation Questions (one per line)", "What is the capital of France?\nWho wrote Hamlet?")
    gt_input = st.text_area("Ground Truths (one per line)", "Paris is the capital of France.\nWilliam Shakespeare wrote Hamlet.")
    
    if st.button("Run Optimizer"):
        questions = [q.strip() for q in q_input.split('\n') if q.strip()]
        ground_truths = [gt.strip() for gt in gt_input.split('\n') if gt.strip()]
        
        if len(questions) != len(ground_truths):
            st.error("Number of questions must match number of ground truths.")
        else:
            with st.spinner("Running deep evaluation across configurations. This may take a while..."):
                try:
                    res = requests.post(
                        f"{API_URL}/optimize",
                        json={"questions": questions, "ground_truths": ground_truths}
                    )
                    if res.status_code == 200:
                        st.success("Optimization Complete!")
                        st.json(res.json()["best_config"])
                    else:
                        st.error(res.text)
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "Metrics":
    st.header("Evaluation Metrics")
    st.write("View the results of the last pipeline optimization run.")
    
    try:
        res = requests.get(f"{API_URL}/metrics")
        if res.status_code == 200:
            data = res.json()
            if "message" in data:
                st.info(data["message"])
            else:
                st.subheader("Best Configuration")
                st.json(data["best_config"])
                
                st.subheader("All Results")
                if "all_results" in data:
                    flat_results = []
                    for r in data["all_results"]:
                        flat = {**r["config"], **r.get("metrics", {}), "composite_score": r["composite_score"]}
                        flat_results.append(flat)
                    st.dataframe(pd.DataFrame(flat_results))
    except Exception as e:
        st.error(f"Connection failed: {e}")
