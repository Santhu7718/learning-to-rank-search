import streamlit as st
import pandas as pd
from src.data_loader import parse_letor_file
from src.data_cleaning import clean_letor_data
import pickle
import os

st.title("Learning-to-Rank Search Demo")
test_path = st.text_input("Test file path", "dataset/MQ2008/Fold1/test.txt")
model_path = st.text_input("Trained model file", "model_fold1.pkl")

if os.path.exists(test_path) and os.path.exists(model_path):
    df = clean_letor_data(parse_letor_file(test_path))
    st.success(f"Loaded {len(df)} rows")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    qids = df["qid"].unique()
    qid = st.selectbox("Select Query ID", qids)
    sub = df[df["qid"] == qid]
    features = [col for col in sub.columns if col.startswith("feat_")]
    sub = sub.copy()
    sub["score"] = model.predict(sub[features])
    st.dataframe(sub.sort_values("score", ascending=False))
else:
    st.info("Provide valid paths to both the test data and saved model (.pkl) file.")
