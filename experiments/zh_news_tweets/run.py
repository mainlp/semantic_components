from semantic_components.sca import SCA
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os
import pickle
import time
import torch
import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)
debug = False

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device available for running: ", device)

store_path = "results/"

# load a dataset, e.g. a sample of tweets
# this DataFrame should have a column "text" with the text of the documents
documents = pd.read_csv("../../data/zh_newspaper_tweets_250924.csv")
print(documents.head())

documents["text_preprocessed"] = documents["text"].replace(
     r'http\S+', '', regex=True).replace(     # remove urls
     r'www\S+', '', regex=True).replace(      # remove other web addresses
     r'\S+\.\S+', '', regex=True).replace(    # remove strings that contain a dot (e.g. emails  and other weird urls not
                                              # caught before)
     r'@\S+', '', regex=True)                 # remove usernames

# exclude tweets with less than 10 characters
documents = documents[documents["text_preprocessed"].apply(lambda x: len(x) >= 20)]

print(len(documents))

if debug:
    documents = documents.sample(3000)

model_name = "paraphrase-multilingual-mpnet-base-v2" #  #"Davlan/afro-xlmr-large"

# check if we have embeddings in temp
if os.path.isfile("embeddings/embeddings_zh_tweets_pp_mpnet_v2.pkl") and not debug:
    with open('embeddings/embeddings_zh_tweets_pp_mpnet_v2.pkl', 'rb') as fp:
        embeddings = pickle.load(fp)
        embeddings = np.array(embeddings)
else:
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)
    print(documents)
    embeddings = embedding_model.encode(documents["text"].tolist(),
                                        show_progress_bar=True, device=device, batch_size=512)
    if not debug:
        with open('embeddings/embeddings_zh_tweets_pp_mpnet_v2.pkl', 'wb') as fp:
            pickle.dump(embeddings, fp)

start_time = time.time()

normalize_components = True
decompositions = "mu"
mu = 1.0
termination_criteria = ["residual_norm", "new_components_history", "max_iterations"]

mcs = 300
ms = 300
max_iterations = 10

if debug:
    mu = 1.0
    mcs = 20
    ms = 20
    max_iterations = 3

# init and run SCA
name = store_path + "sca_normco_N={}_mu={}_mcs={}_ms={}_ng".format(len(documents), mu, mcs, ms)

sca = SCA(
    hdbscan_min_cluster_size=mcs,
    hdbscan_min_samples=ms,
    normalize_components=True,
    max_iterations=max_iterations,
    mu=mu,
    alpha_decomposition=0.1,
    n_grams=3,
    stopwords_path="../../data/stopwords_hau.txt",
    verbose=True,
    logging=True,
    language="zh",
    termination_crit=termination_criteria,
    evaluation=True
    )
scores, residuals, ids = sca.fit_transform(documents, embeddings)

time_elapsed = time.time() - start_time

print("Time elapsed: ", time_elapsed)

str = sca.get_representation_string()

# to file
if not os.path.exists(name):
    os.makedirs(name)

with open(name + "/representation.txt", "w") as f:
    f.write(str)

# save representations df
sca.representations.to_pickle(name + "/representations.pkl")

# save results
sca.save(name)