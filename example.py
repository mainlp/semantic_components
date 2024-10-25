from semantic_components.sca import SCA
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

## preprocess documents
documents = pd.read_csv("data/trump_tweets.csv")
documents["text_preprocessed"] = documents["text"].replace(
    r'http\S+', '', regex=True).replace(     # remove urls
    r'www\S+', '', regex=True).replace(      # remove other web addresses
    r'\S+\.\S+', '', regex=True).replace(    # remove strings that contain a dot (e.g. emails  and other weird urls not
                                             # caught before)
    r'@\S+', '', regex=True)                 # remove usernames
# exclude tweets with less than 30 characters
documents = documents[documents["text_preprocessed"].apply(lambda x: len(x) >= 30)]

## create results folder
if not os.path.exists("results/"):
    os.makedirs("results/")

if not os.path.exists("results/embeddings/"):
    os.makedirs("results/embeddings/")

## embed texts (or load previously computed embeddings)
if os.path.isfile("results/embeddings/embeddings_trump_pp_mpnet_v2.pkl"):
    with open('results/embeddings/embeddings_trump_pp_mpnet_v2.pkl', 'rb') as fp:
        embeddings = pickle.load(fp)
        embeddings = np.array(embeddings)
else:
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)
    embeddings = embedding_model.encode(documents["text"].tolist(), 
                                        show_progress_bar=True, device=device, batch_size=512)

    with open('results/embeddings/embeddings_trump_pp_mpnet_v2.pkl', 'wb') as fp:
        pickle.dump(embeddings, fp)

## run SCA (it's just two lines of code!)
sca = SCA(
    alpha_decomposition=0.2, 
    mu=0.95, 
    hdbscan_min_cluster_size=100,
    hdbscan_min_samples=50,
    combine_overlap_threshold=0.5)
scores, residuals, ids = sca.fit(documents, embeddings)

## save results
representations = sca.representations  # pandas df
representations.to_csv("results/representations_trump_pp_mpnet_v2.csv", index=True)

print(representations)