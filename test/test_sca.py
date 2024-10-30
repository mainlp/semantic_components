import unittest
import pandas as pd
import numpy as np

# from cluster_pipeline import CTFIDFInterface
from semantic_components.sca import SCA


class ToyHDBSCAN:
    def __init__(self, min_cluster_size, min_samples):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.labels_ = np.array([0, -1, 0, -1])
        self.probabilities_ = np.array([1, 0.5, 0.5, 1])

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array([0, -1, 0, -1]), np.array([1, 0.5, 0.5, 1])

    def fit_predict(self, X):
        return np.array([0, -1, 0, -1]), np.array([1, 0.5, 0.5, 1])

    def get_params(self, deep=True):
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
        }


class ToyUMAP:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [[0.1, 1.2], [1.9, 0.5], [-1.1, -1], [2.3, 0.1]][: len(X)]
        )

    def fit_transform(self, X):
        return np.array(
            [[0.1, 1.2], [1.9, 0.5], [-1.1, -1], [2.3, 0.1]][: len(X)]
        )

    def get_params(self, deep=True):
        return {"n_components": self.n_components}


class ToyEmbedding:
    def __init__(self):
        pass

    def encode(self, documents, show_progress_bar=False):
        return np.array(
            [
                [0, 1, 0, 0.2, 1.2],
                [1, 0, 2, 0.75, 0.3],
                [2, -1, -1, 0.75, 0.1],
                [-1, 2, 1, 1.25, 2],
            ]
        )

    def get_params(self, deep=True):
        return {}


class TestSCA(unittest.TestCase):

    def setUp(self):
        # create a sample dataset
        self.documents = pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "text": [
                    "This is the first document." * 10,
                    "This is the second document." * 10,
                    "And this is the third one." * 10,
                    "And this is the fourth." * 10,
                ],
                "text_preprocessed": [
                    "This is the first document." * 10,
                    "This is the second document." * 10,
                    "And this is the third one." * 10,
                    "And this is the fourth." * 10,
                ],
                "cluster_id": [0, 0, 1, 1],
            }
        )
        self.ground_truth = np.array([0, 0, 1, -1])
        self.embeddings = np.array(
            [
                [0, 1, 0, 0.2, 1.2],
                [1, 0, 2, 0.75, 0.3],
                [2, -1, -1, 0.75, 0.1],
                [-1, 2, 1, 1.25, 2],
            ]
        )
        self.clustering_algorithm = ToyHDBSCAN(
            min_cluster_size=2, min_samples=1
        )
        self.embedding_model = ToyEmbedding()
        self.dim_reduction = ToyUMAP(n_components=2)

    def test_decompose(self):
        sca = SCA(
            cluster_algorithm=self.clustering_algorithm,
            dim_reduction_algorithm=self.dim_reduction,
            normalize_components=True,
            mu=0.5,
            termination_crit="new_components",
        )
        residuals, scores, ids = sca.decompose(self.documents, self.embeddings)
        print(residuals)
        print(scores)
        print(ids)

    def test_fit_transform_strong(self):
        sca = SCA(
            dim_reduction_algorithm=self.dim_reduction,
            cluster_algorithm=self.clustering_algorithm,
            normalize_components=True,
            verbose=True,
            termination_crit="new_components",
        )

        scores, residuals, ids = sca.fit_transform(
            self.documents, self.embeddings
        )
        scores_t, residuals_t, ids_t = sca.transform(self.embeddings)

        print("Scores:", scores, scores_t)

        # self.assertTrue((scores == scores_t).all())
        # self.assertTrue((residuals == residuals_t).all())

        print(scores)
        print(residuals)

    def test_fit_transform_weak(self):
        sca = SCA(
            dim_reduction_algorithm=self.dim_reduction,
            cluster_algorithm=self.clustering_algorithm,
            normalize_components=True,
            mu=0.5,
            termination_crit="new_components",
        )

        result = sca.fit_transform(self.documents, self.embeddings)
        print(result)

    def test_represent_ie_fit(self):
        sca = SCA(
            dim_reduction_algorithm=self.dim_reduction,
            cluster_algorithm=self.clustering_algorithm,
            normalize_components=True,
            mu=0.5,
            termination_crit="new_components",
        )

        result = sca.fit(
            self.documents,
            self.embeddings,
        )
        print(result)
        print(sca.representations)
        self.assertEqual(len(sca.representations) - 1, result[0].shape[1])

    def test_alpha_fit(self):
        sca = SCA(
            dim_reduction_algorithm=self.dim_reduction,
            cluster_algorithm=self.clustering_algorithm,
            normalize_components=True,
            mu=0.5,
            alpha_decomposition=0.1,
            termination_crit="new_components",
        )

        result = sca.fit(self.documents, self.embeddings)
        print(result)
        print(sca.representations)
        self.assertEqual(len(sca.representations) - 1, result[0].shape[1])

    def test_evaluate(self):
        sca = SCA(
            dim_reduction_algorithm=self.dim_reduction,
            cluster_algorithm=self.clustering_algorithm,
            normalize_components=True,
            evaluation=True,
            mu=0.5,
            termination_crit="new_components",
            verbose=True,
        )

        sca.fit(self.documents, self.embeddings)
