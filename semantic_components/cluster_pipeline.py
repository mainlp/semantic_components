"""
A pipeline for realizing our clustering experiments. This class is used to
abstract from boilerplate code that is reused throughout experiments. The
class is initialized with a clustering algorithm, a dataset of documents
and optionally precomputed embeddings as well as a dimensionality reduction.
To interpret results, a variety of representation and evaluation techniques
is provided.
"""

import numpy as np
import collections
from hdbscan import HDBSCAN
from umap import UMAP


class EmptyUMAP:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X


class ClusterPipeline:
    """
    A pipeline for realizing our clustering experiments. This class is used to
    abstract from boilerplate code that is reused throughout experiments. The
    class is initialized with a clustering algorithm, a dataset of documents
    and optionally precomputed embeddings as well as a dimensionality reduction.
    To interpret results, a variety of representation and evaluation techniques
    is provided.

    The pipeline is structured as follows:

    1. Embed the documents (skip if embeddings are provided)
    2. Dimensionality reduction (skip if none is provided)
    3. Clustering
    4. Evaluation and Representation
    """

    def __init__(
        self,
        cluster_algorithm,
        dim_reduction,
        verbose=False,
        report=print,
        log=None,
        name="cluster_pipeline",
    ):
        """
        Initialize the pipeline with a clustering algorithm, a dataset and optionally
        precomputed embeddings as well as a dimensionality reduction.
        """
        self.cluster_algorithm = cluster_algorithm
        self.dim_reduction = dim_reduction

        self.vocab = None
        self.stopwords = None

        self.vectorizer = None
        self.name = name
        self.stopwords = []
        self.verbose = verbose
        self.ctfidf_interface = None
        self.log = lambda x, y: None if log is None else log(x, y)
        self.report = report

        if self.verbose:

            self.report(
                "==========================INITIALIZATION============================"
            )
            self.report("Clustering algorithm {}", [str(type(cluster_algorithm))])
            if type(cluster_algorithm) is HDBSCAN:
                self.report(
                    "Clustering algorithm params: min_cluster_size={}, min_samples={}",
                    [
                        cluster_algorithm.min_cluster_size,
                        cluster_algorithm.min_samples,
                    ],
                )
            if type(dim_reduction) is not None:
                self.report(
                    "Dimensionality reduction {}", [str(type(self.dim_reduction))]
                )
                if type(self.dim_reduction) is UMAP:
                    self.report(
                        "Dimensionality reduction params: n_neighbors={}, min_dist={}",
                        [
                            self.dim_reduction.n_neighbors,
                            self.dim_reduction.min_dist,
                        ],
                    )

    def fit(self):
        """
        Fit the pipeline.
        """
        return self.fit_transform()

    def fit_transform(self, embeddings):
        """
        Run the pipeline. This will embed the documents, reduce its dimensionality
        and cluster it. The results are then evaluated and represented.
        """
        if embeddings is None:
            raise ValueError("ClusterPipeline: No embeddings provided.")

        if self.dim_reduction is not None:
            reduced = self.dim_reduction.fit_transform(embeddings)
            reduced = np.array(reduced)

        else:
            self.report("No dimensionality reduction specified.")

        clustered = self.cluster_algorithm.fit_predict(reduced)

        if type(clustered) is tuple and len(clustered) == 2:
            ids, probs = clustered
        elif type(clustered) == np.ndarray and (len(clustered.shape) == 1):
            ids = clustered
            probs = self.cluster_algorithm.probabilities_
        elif type(clustered) is list:
            ids = np.array(clustered)
            probs = self.cluster_algorithm.probabilities_
        else:
            try:
                ids = np.array(clustered)
                probs = self.cluster_algorithm.probabilities_
            except Exception as e:
                raise ValueError(
                    "Clustering algorithm did not return expected output."
                )

        ids = self._ordered_component_ids(ids)

        if max(ids) == -1:
            self.report("ClusterPipeline: No clusters found.")

        self.log(f"component_ids_{self.name}", ids)
        self.log(f"cluster_probs_{self.name}", probs)

        return ids, probs

    def _ordered_component_ids(self, ids):
        """
        Reassign the cluster ids in decreasing size of their respective cluster.

        In other words: Make cluster with id 0 the largest cluster, cluster with id 1
        the second largest, etc.
        """
        counter = collections.Counter(ids)
        unique_ids = np.unique(ids).tolist()

        if -1 in unique_ids:
            unique_ids.remove(-1)
        sorted_ids = sorted(unique_ids, key=lambda x: counter[x], reverse=True)

        id_to_new_id = {sorted_ids[i]: i for i in range(len(sorted_ids))}
        id_to_new_id[-1] = -1

        return np.array([id_to_new_id[id] for id in ids])

    def get_number_of_clusters(self, ids):
        """
        Compute the number of topics in the clustering.
        """
        return len(np.unique(ids))
