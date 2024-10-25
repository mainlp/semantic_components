from semantic_components.sca import SCA
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import pickle
import time
import torch
import sys

import warnings
warnings.filterwarnings("ignore")

debug = False



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device available for running: ", device)


def run_sca(
        mcs=50,
        ms=50,
        alpha=0.0,
        mu=0.9,
        debug=False,
        seed=0,
        prefix=""):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load a dataset, e.g. a sample of tweets
    # this DataFrame should have a column "text" with the text of the documents
    documents = pd.read_csv("../../data/trump_tweets.csv")

    documents["text_preprocessed"] = documents["text"].replace(
        r'http\S+', '', regex=True).replace(     # remove urls
        r'www\S+', '', regex=True).replace(      # remove other web addresses
        r'\S+\.\S+', '', regex=True).replace(    # remove strings that contain a dot (e.g. emails  and other weird urls 
                                                 # not caught before)
        r'@\S+', '', regex=True)                 # remove usernames

    os.makedirs("embeddings", exist_ok=True)
    os.makedirs(prefix + "results", exist_ok=True)

    # exclude tweets with less than 30 characters
    documents = documents[documents["text_preprocessed"].apply(lambda x: len(x) >= 30)]
    
    if debug:
        documents = documents.sample(3000)

    store_path = prefix + "results/"
    name = store_path + "sca_grid_run_N={}_mu={}_alpha={}_mcs={}_ms={}_seed={}".format(len(documents), 
                                                                                       mu, alpha, mcs, ms, seed)

    if not os.path.exists(name):
        os.makedirs(name)
    else:
        print(name)
        print("This config was already run. Skipping.")
        return None

    model_name = "paraphrase-multilingual-mpnet-base-v2"

    if model_name == "paraphrase-multilingual-mpnet-base-v2":

        # check if we have embeddings in temp
        if os.path.isfile("embeddings/embeddings_trump_pp_mpnet_v2.pkl") and not debug:
            with open('embeddings/embeddings_trump_pp_mpnet_v2.pkl', 'rb') as fp:
                embeddings = pickle.load(fp)
                embeddings = np.array(embeddings)
        else:
            embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)
            print(documents)
            embeddings = embedding_model.encode(documents["text"].tolist(), 
                                                show_progress_bar=True, device=device, batch_size=512)
            if not debug:
                with open('embeddings/embeddings_trump_pp_mpnet_v2.pkl', 'wb') as fp:
                    pickle.dump(embeddings, fp)
    else:
        raise ValueError("Model not implemented for this use case.")
    
    start_time = time.time()

    max_iterations = 10
    termination_criteria = ["residual_norm", "new_components_history", "max_iterations"]

    if debug:
        mu = 1.0
        mcs = 20
        ms = 20
        max_iterations = 3

    # init and run SCA
    sca = SCA(
        hdbscan_min_cluster_size=mcs,
        hdbscan_min_samples=ms,
        normalize_components=True,
        max_iterations=max_iterations,
        mu=mu,
        alpha_decomposition=alpha,
        n_grams=3,
        stopwords_path="../../data/stopwords_hau.txt",
        verbose=True,
        logging=True,
        language="other",
        termination_crit=termination_criteria,
        evaluation=True
        )
    scores, residuals, ids = sca.fit_transform(documents, embeddings)

    print("Time elapsed: ", time.time() - start_time)

    str = sca.get_representation_string()

    # to file
    with open(name + f"/representations@{time.asctime()}.txt", "w") as f:
        f.write(str)

    # save representations df
    sca.representations.to_pickle(name + f"/representations@{time.asctime()}.pkl")
    # save results
    sca.save(name)

    if not debug:

        # find the tweet
        tweet = "With the exception of New York"
        tweet_id = documents[documents["text"].str.startswith(tweet)]
        tweet_id = tweet_id.index[0]

        # get score
        embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)
        embedding = embedding_model.encode([documents.loc[tweet_id]["text_preprocessed"]], 
                                           show_progress_bar=True, device=device, batch_size=1)
        scores, residuals = sca.get_scores(embedding)

        # join with representations and sort
        df_scores = sca.representations.copy()
        df_scores["score"] = [-1] + np.reshape(scores, -1).tolist()
        df_scores = df_scores.sort_values(by="score", ascending=False)

        expl = ""

        # get top 10 for each iteration
        for i in range(0, min(len(df_scores["iteration"].unique()) - 1, 10)):
            top_10 = df_scores[df_scores["iteration"] == i].head(10)

            for i in range(0, min(10, len(top_10))): 
                expl += f"name: {top_10.iloc[i]['name']}  iteration: {top_10.iloc[i]['iteration']} \n"
                expl += f"representation: {top_10.iloc[i]['representation']} \n"
                expl += f"medoid: {top_10.iloc[i]['representation_medoid']} \n"
                expl += f"score: {top_10.iloc[i]['score']} \n\n"

        # store explanation
        with open(name + f"/explanation@{time.asctime()}.txt", "w") as f:
            f.write(expl)

    return documents, embeddings, sca
        

if __name__ == "__main__":
    # read args from command line
    mcs = int(sys.argv[1])
    ms = int(sys.argv[2])
    alpha = float(sys.argv[3])
    mu = float(sys.argv[4])
    if len(sys.argv) > 5:
        seed = int(sys.argv[5])
    else:
        seed = 0
    print(f"Running SCA with mcs={mcs}, ms={ms}, alpha={alpha}, mu={mu}, seed={seed}")
    run_sca(mcs=mcs, ms=ms, alpha=alpha, mu=mu, debug=False, seed=seed)