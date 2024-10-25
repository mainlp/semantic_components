
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
torch.manual_seed(0)

debug = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device available for running: ", device)

os.makedirs("embeddings", exist_ok=True)
os.makedirs("results", exist_ok=True)

store_path = "results/"

# load a dataset, e.g. a sample of tweets
# this DataFrame should have a column "text" with the text of the documents
documents = []
for lang in ["hau"]:
    with open("../../data/nigerian_tweets/{}.txt".format(lang), "r") as f:
        for line in f:
            documents.append((line, lang))
documents = pd.DataFrame(documents, columns=["text", "lang"])

# exclude tweets with less than 20 characters
documents = documents[documents["text"].apply(lambda x: len(x) >= 30)]

if debug:
    documents = documents.sample(3000)

model_name = "sentence-transformers/LaBSE" # "paraphrase-multilingual-mpnet-base-v2" #  #"Davlan/afro-xlmr-large"

if model_name == "paraphrase-multilingual-mpnet-base-v2" and not debug:

    # check if we have embeddings in temp
    if os.path.isfile("embeddings/embeddings_pp_mpnet_v2.pkl") and not debug:
        with open('embeddings/embeddings_pp_mpnet_v2.pkl', 'rb') as fp:
            embeddings = pickle.load(fp)
            embeddings = np.array(embeddings)
    else:
        embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2").to(device)
        print(documents)
        embeddings = embedding_model.encode(documents["text"].tolist(),
                                             show_progress_bar=True, device=device, batch_size=512)
        if not debug:
            with open('embeddings/embeddings_pp_mpnet_v2.pkl', 'wb') as fp:
                pickle.dump(embeddings, fp)

elif model_name == "sentence-transformers/LaBSE" or debug:
    
        # check if we have embeddings in temp
        if os.path.isfile("embeddings/embeddings_labse.pkl") and not debug:
            with open('embeddings/embeddings_labse.pkl', 'rb') as fp:
                embeddings = pickle.load(fp)
                embeddings = np.array(embeddings)
        else:
            embedding_model = SentenceTransformer("sentence-transformers/LaBSE").to(device)
            embeddings = embedding_model.encode(documents["text"].tolist(),
                                                 show_progress_bar=True, device=device, batch_size=512)
            if not debug:
                with open('embeddings/embeddings_labse.pkl', 'wb') as fp:
                    pickle.dump(embeddings, fp)

elif model_name == "Davlan/afro-xlmr-large":
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large")
    model = AutoModelForMaskedLM.from_pretrained("Davlan/afro-xlmr-large").to(device)

    # check if we have embeddings in temp
    if os.path.isfile("embeddings/embeddings_afro_xlmr_large.pkl"):
        with open('embeddings/embeddings_afro_xlmr_large.pkl', 'rb') as fp:
            embeddings = pickle.load(fp)
            embeddings = np.array(embeddings)
    else:
        embeddings = []
        # get dataloader of documents for batching
        dataloader = torch.utils.data.DataLoader(documents["text"].tolist(), batch_size=128, shuffle=False)
        for i, batch in enumerate(dataloader):
            print("Batch: ", i)
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: inputs[key].to(device) for key in inputs}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            print(len(outputs.hidden_states))
            print(outputs.hidden_states[-2].shape)
            embeddings.append(outputs.logits.cpu().numpy())
        embeddings = np.concatenate(embeddings)
        with open('embeddings/embeddings_afro_xlmr_large.pkl', 'wb') as fp:
            pickle.dump(embeddings, fp)


start_time = time.time()

normalize_components = True
normalize_residuals = False
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
    language="other",
    termination_crit=termination_criteria,
    evaluation=True
    )
scores, residuals, ids = sca.fit_transform(documents, embeddings)

str = sca.get_representation_string()

# to file
with open(name + ".txt", "w") as f:
    f.write(str)

# save representations df
sca.representations.to_pickle(name + "/representations.pkl")

# save results
sca.save(name)

print("Time elapsed: ", time.time() - start_time)