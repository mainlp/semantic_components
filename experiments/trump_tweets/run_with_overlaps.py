from run_fct import run_sca
import os
import re
import pandas as pd



def get_cluster_hierarchy(sca):
    # get hdbscan of first iteration
    first_decomposer = sca.decomposer.decomposers[0]
    clusterer = first_decomposer.cluster_pipeline.cluster_algorithm


    tree = clusterer.condensed_tree_.to_pandas()
    graph = clusterer.condensed_tree_.to_networkx()


    all_clusters = tree[tree.child_size >= clusterer.min_cluster_size]
    all_clusters_set = set(all_clusters.child.tolist() + all_clusters.parent.tolist())

    # compute the assignments of each document
    assignments = []
    for i in range(0, len(documents)):
        pred = [i]
        while [p for p in graph.predecessors(pred[-1])]:
            # since its a tree, we have only one predecessor for each node
            pred += [p for p in graph.predecessors(pred[-1])]
        
        pred = [p for p in pred if p in all_clusters_set]  # remove non-cluster nodes
        if len(pred) == 0:
            print("Has no assignment:", i)

        assignments.append(pred)

    return assignments

def calculate_overlap(documents, component_col="component_id", compare_col="hierarchy_assignment"):
    documents_ex1 = documents.explode(component_col)
    components = documents_ex1.groupby(component_col).agg(count=(compare_col, "count"))
    documents_ex2 = documents.explode(compare_col)
    nodes = documents_ex2.groupby(compare_col).agg(count=(component_col, "count"))
    documents = documents_ex1.explode(compare_col)

    # remove -1 from components
    components = components[components.index != -1]

    overlaps = []
    for component in components.index:
        component_samples = documents[documents[component_col] == component]
        grouped = component_samples.groupby(compare_col).agg(count=(component_col, "count"))
        grouped["overlap"] = grouped["count"] / (
            nodes.loc[grouped.index]["count"] + components.loc[component]["count"] - grouped["count"])
        max_overlap = grouped["overlap"].max()
        overlaps.append(max_overlap)
    
    components["max_overlap"] = overlaps

    return components


mcs = 100
ms = 50
alpha = 0.2
mu = 0.95
debug = False
seed = 0

documents, embeddings, sca = run_sca(mcs, ms, alpha, mu, debug, seed)
assignments = get_cluster_hierarchy(sca)

for i, row in sca.representations.iterrows():
    if row["combined_id"] != i:
        print(i, "merged to", row["combined_id"])
        print(row["representation"])
        print(sca.representations.loc[row["combined_id"]]["representation"])
    
documents["hierarchy_assignment"] = assignments

components = calculate_overlap(documents)

# load representations and add overlap score
path = f"results/sca_grid_run_N=50343_mu={mu}_alpha={alpha}_mcs={mcs}_ms={ms}_seed={seed}/"
files = os.listdir(path)
repr_fname = [f for f in files if re.findall(r".*representations@.*\.pkl", f)][0]
repr = pd.read_pickle(path + repr_fname)

repr["max_overlap_bertopic"] = [e for e in map(
    lambda x: components.loc[x]["max_overlap"] if x in components.index else -1, repr.index.to_list())]

# calculate token overlap scores
first_iteration = repr[repr["iteration"] == 0]
token_overlaps = []
for i in range(len(repr)):
    overlaps_with_first = []
    for j in range(len(first_iteration)):
        overlaps_with_first.append(len(set(repr.iloc[i]["representation"]).intersection(first_iteration.iloc[j]["representation"])))
    token_overlaps.append(max(overlaps_with_first))


repr["max_token_overlap_bertopic"] = token_overlaps

# save
repr.to_pickle(path + repr_fname)

# calculate averages and append to reports
avg_overlap = repr[(repr["iteration"] > 0) & (repr["max_overlap_bertopic"] >= 0)]["max_overlap_bertopic"].mean()
avg_token_overlap = repr[
    (repr["iteration"] > 0) & (repr.index==repr["combined_id"])]["max_token_overlap_bertopic"].mean()

report_fname = [f for f in files if re.match(r".*reports@.*\.txt", f)][0]

with open(path + report_fname, "a") as f:
    f.write("\n")
    f.write(f"Average cluster overlap with first iteration: {avg_overlap}\n")
    f.write(f"Average token overlap with fist iteration: {avg_token_overlap}\n")
    f.write("\n")