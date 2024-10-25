import os
import sys

from semantic_components.sca import SCA
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn

from mteb import MTEB, get_benchmark


def run_sca(embeddings, documents, debug=False):

    name = "sca"

    n_it = 10

    if debug:
        n_it = 3

    # init and run SCA
    sca = SCA(
        hdbscan_min_cluster_size=50,
        hdbscan_min_samples=50,
        normalize_components=True,
        max_iterations=n_it,
        mu=0.9,
        alpha_decomposition=0.0,
        n_grams=3,
        name=name,
        stopwords_path="../../data/stopwords_hau.txt",
        verbose=True,
        logging=False,
        language="other",
        termination_crit=["residual_norm", "new_components_history", "max_iterations"],
        evaluation=False
        )
    scores, residuals, ids = sca.fit_transform(documents, embeddings)

    return sca

def get_data_and_model(debug=False, device="cuda"):
    # run a trump sca
    documents = pd.read_csv("../../data/trump_tweets.csv")

    documents["text_preprocessed"] = documents["text"].replace(
        r'http\S+', '', regex=True).replace(     # remove urls
        r'www\S+', '', regex=True).replace(      # remove other web addresses
        r'\S+\.\S+', '', regex=True).replace(    # remove strings that contain a dot (e.g. emails  and other weird urls
                                                 # not caught before)
        r'@\S+', '', regex=True)                 # remove usernames

    # exclude tweets with less than 30 characters
    documents = documents[documents["text_preprocessed"].apply(lambda x: len(x) >= 30)]

    if debug:
        documents = documents.sample(3000)

    print(len(documents))
 
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)

    # check if we have embeddings in temp
    if os.path.isfile("embeddings/embeddings_trump_pp_mpnet_v2.pkl") and not debug:
        with open('embeddings/embeddings_trump_pp_mpnet_v2.pkl', 'rb') as fp:
            embeddings = pickle.load(fp)
            embeddings = np.array(embeddings)
    else:
        print(documents)
        embeddings = embedding_model.encode(documents["text"].tolist(),
                                            show_progress_bar=True, device=device, batch_size=512)
        if not debug:
            with open('embeddings/embeddings_trump_pp_mpnet_v2.pkl', 'wb') as fp:
                pickle.dump(embeddings, fp)

    return embeddings, documents, embedding_model


def run_pca(embeddings, documents, dim=2):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=dim)
    pca.fit(embeddings)

    return pca


class EmbeddingPipelineSCA(nn.Module):
    """
    A wrapper for providing an embedding model to MTEB. It consists of the base
    embedding model as well as a SCA transformation on top.
    """

    def __init__(self, sbert_model, sca_transform, add_docs=False, device="cuda", top_n=-1):
        super(EmbeddingPipelineSCA, self).__init__()

        self.device = device
        self.add_docs = add_docs
        self.top_n = top_n

        self.sbert_model = sbert_model
        self.sca_transform = sca_transform

    def forward(self, x):
        with torch.no_grad():
            out = self.encode(x)
        return out

    def encode(self, x, batch_size=256, **kwargs):
        emb = self.sbert_model.encode(x, batch_size=batch_size)
        transformed, residuals = self.sca_transform.get_scores(emb, top_n=self.top_n)
        return transformed
    

class EmbeddingPipeline(nn.Module):
    """
    A wrapper for providing an embedding model to MTEB. It consists of the base
    embedding model as well as a SCA transformation on top.
    """

    def __init__(self, sbert_model, sca_transform, add_docs=False, device="cuda"):
        super(EmbeddingPipeline, self).__init__()

        self.device = device
        self.add_docs = add_docs

        self.sbert_model = sbert_model
        self.sca_transform = sca_transform

    def forward(self, x):
        with torch.no_grad():
            out = self.encode(x)
        return out

    def encode(self, x, batch_size=256, **kwargs):
        emb = self.sbert_model.encode(x, batch_size=batch_size)
        transformed = self.sca_transform.transform(emb)
        return transformed
    

def mteb_eval(model, path, device="cuda", debug=False, order=1):

    if debug:
        path += "/debug"

    mteb_tasks=get_benchmark("MTEB(eng)")
    if debug:
        mteb_tasks = ["STS12"]

    mteb_tasks = mteb_tasks[::order]

    evaluation = MTEB(tasks=mteb_tasks)
    results_sbert = evaluation.run(model, output_folder=path, device=device)
    return results_sbert


if __name__ == "__main__":

    device = "cuda"
    if len(sys.argv) == 3:
        device = sys.argv[2]

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Device available for running: ", device)


    debug=False
    if len(sys.argv) > 1:
        debug=sys.argv[1] == "debug"

    print("Running SCA on Trump tweets")
    embeddings, documents, embedding_model = get_data_and_model(debug=debug, device=device)
    sca = run_sca(embeddings, documents, debug=True)
    print("Finished running SCA")

    dim_first = sca.decomposer.n_comp_history[0]
    dim = len(sca.components)
    print(f"Training PCAs of dimensions {dim_first} and {dim}")

    pca_first = run_pca(embeddings, documents, dim=dim_first)
    pca = run_pca(embeddings, documents, dim=dim)

    print("Finished running PCAs")

    sca_embedding = EmbeddingPipelineSCA(embedding_model, sca, device=device)
    sca_first_embedding = EmbeddingPipelineSCA(embedding_model, sca, device=device, top_n=dim_first)
    pca_embedding = EmbeddingPipeline(embedding_model, pca, device=device)
    pca_first_embedding = EmbeddingPipeline(embedding_model, pca_first, device=device)

    start = 0
    if len(sys.argv) > 1 and not debug:
        i=int(sys.argv[1])

    if len(sys.argv) > 3 and not debug:
        order = int(sys.argv[3])

    if i <= 0:
        print("Running MTEB evaluation on SCA")
        print(mteb_eval(sca_embedding, "results/sca_embedding", debug=debug, order=order))
    if i <= 1:
        print("Running MTEB evaluation on SCA first")
        print(mteb_eval(sca_first_embedding, "results/sca_first_embedding", debug=debug, order=order))
    if i <= 2:
        print("Running MTEB evaluation on PCA")
        print(mteb_eval(pca_embedding, "results/pca_embedding", debug=debug, order=order))
    if i <= 3:
        print("Running MTEB evaluation on PCA first")
        print(mteb_eval(pca_first_embedding, "results/pca_first_embedding", debug=debug, order=order))
    if i <= 4:
        print("Running MTEB evaluation on vanilla SBERT")
        print(mteb_eval(embedding_model, "results/vanilla_embedding", debug=debug, order=order))
    

