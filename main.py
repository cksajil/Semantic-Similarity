import numpy as np
import pandas as pd
import streamlit as st
from numpy.linalg import norm
from tqdm import tqdm
from os.path import join
from sentence_transformers import SentenceTransformer


def load_df():
    df_path = join("static", "df2.csv")
    df = pd.read_csv(df_path, index_col=False).dropna().iloc[:, 1:]
    return df


def create_embeddings():
    embd_matrix = []
    embd_path = join("static", "pretrained_embeddings.npy")
    df = load_df()
    model = SentenceTransformer("bert-base-nli-mean-tokens")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        embd_matrix.append(model.encode(row["text"]))
    embd_matrix = np.array(embd_matrix)

    with open(embd_path, "wb") as f:
        np.save(f, embd_matrix)


def load_embeddings():
    embd_path = join("static", "pretrained_embeddings.npy")
    embd_matrix = np.load(embd_path)
    return embd_matrix


def get_top5_embedding_rows(user_input):
    df = load_df()
    model = SentenceTransformer("bert-base-nli-mean-tokens")
    similarity_vector = []

    vect2 = model.encode(user_input)
    embd_matrix = load_embeddings()

    for emb_row in tqdm(embd_matrix):
        vect1 = emb_row
        cos_sim = np.dot(vect1, vect2) / (norm(vect1) * norm(vect2))
        similarity_vector.append(cos_sim)

    similarity_vector = np.array(similarity_vector)
    top5_indx = np.argpartition(similarity_vector, -5)[-5:]

    df["similarity"] = similarity_vector

    return df.iloc[top5_indx]


# create_embeddings()

st.title("Semantic similarity")
text_in = st.text_input(
    "Type review text here", "This is user input from NetFlix"
).lower()

if st.button("Predict") and text_in:
    result = get_top5_embedding_rows(text_in)
    result["text"] = result["text"].str.lower()
    result = result.sort_values(by=["similarity"], ascending=False)
    st.dataframe(result)
