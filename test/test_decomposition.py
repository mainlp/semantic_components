import unittest
import pandas as pd
from semantic_components.cluster_pipeline import ClusterPipeline
from semantic_components.decomposition import ClusterDecomposer
from semantic_components.decomposition import ResidualDecomposer
import numpy as np

# from evaluation import CTFIDFInterface


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


class TestClusterPipeline(unittest.TestCase):

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
        # self.ground_truth = np.array([0, 0, 1, -1])
        self.embeddings = np.array(
            [
                [0, 1, 0, 0.2, 1.2],
                [1, 0, 2, 0.75, 0.3],
                [2, -1, -1, 0.75, 0.1],
                [-1, 2, 1, 1.25, 2],
            ]
        )
        self.cluster_algorithm = ToyHDBSCAN(min_cluster_size=2, min_samples=1)
        # self.embedding_model = ToyEmbedding()
        self.dim_reduction = ToyUMAP(n_components=2)

    def test_init(self):
        # test initialization of ClusterPipeline without embedding
        pipeline = ClusterPipeline(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
        )

        self.assertIsNotNone(pipeline.dim_reduction)

    def test_fit_transform(self):
        # test initialization of ClusterPipeline without embedding
        pipeline = ClusterPipeline(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
        )
        ids, probs = pipeline.fit_transform(self.embeddings)
        self.assertTrue(len(ids) == len(self.embeddings))
        self.assertTrue(len(probs) == len(self.embeddings))


class TestClusterDecomposer(unittest.TestCase):

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
        # self.ground_truth = np.array([0, 0, 1, -1])
        self.embeddings = np.array(
            [
                [0, 1, 0, 0.2, 1.2],
                [1, 0, 2, 0.75, 0.3],
                [2, -1, -1, 0.75, 0.1],
                [-1, 2, 1, 1.25, 2],
            ]
        )
        self.cluster_algorithm = ToyHDBSCAN(min_cluster_size=2, min_samples=1)
        # self.embedding_model = ToyEmbedding()
        self.dim_reduction = ToyUMAP(n_components=2)

    def test_init(self):
        # test initialization of ClusterPipeline without embedding
        pipeline = ClusterDecomposer(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
        )

        self.assertIsNotNone(pipeline.dim_reduction)

    def test_fit_transform(self):
        # test initialization of ClusterPipeline without embedding
        pipeline = ClusterDecomposer(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
        )
        residuals, scores, ids = pipeline.fit_transform(self.embeddings)

        self.assertTrue(len(ids) == len(self.embeddings))

        residuals_transformed, scores_transformed, _ = pipeline.transform(
            self.embeddings
        )

        print(residuals_transformed, residuals)

        self.assertTrue((residuals == residuals_transformed).all())
        self.assertTrue((scores == scores_transformed).all())

    def test_fit_transform_mu(self):
        # test initialization of ClusterPipeline without embedding
        pipeline = ClusterDecomposer(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
        )
        residuals, scores, ids = pipeline.fit_transform(self.embeddings)

        self.assertTrue(len(ids) == len(self.embeddings))

        residuals_transformed, scores_transformed, _ = pipeline.transform(
            self.embeddings
        )

        print(residuals_transformed, residuals)

        self.assertTrue((residuals == residuals_transformed).all())
        self.assertTrue((scores == scores_transformed).all())

    def test_fit_transform_full_mu(self):
        # test initialization of ClusterPipeline without embedding
        pipeline = ClusterDecomposer(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
            normalize_residuals=True,
            mu=0.8,
        )
        residuals, scores, ids = pipeline.fit_transform(self.embeddings)

        self.assertTrue(len(ids) == len(self.embeddings))

        residuals_transformed, scores_transformed, _ = pipeline.transform(
            self.embeddings
        )

        self.assertTrue((residuals == residuals_transformed).all())
        self.assertTrue((scores == scores_transformed).all())


class TestResidualDecomposer(unittest.TestCase):

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
        # self.ground_truth = np.array([0, 0, 1, -1])
        self.embeddings = np.array(
            [
                [0, 1, 0, 0.2, 1.2],
                [1, 0, 2, 0.75, 0.3],
                [2, -1, -1, 0.75, 0.1],
                [-1, 2, 1, 1.25, 2],
            ]
        )
        self.cluster_algorithm = ToyHDBSCAN(min_cluster_size=2, min_samples=1)
        # self.embedding_model = ToyEmbedding()
        self.dim_reduction = ToyUMAP(n_components=2)

    def test_init(self):
        decomposer = ClusterDecomposer(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
            normalize_residuals=True,
            mu=0.8,
        )
        ResidualDecomposer(decomposer=decomposer)

    def test_fit_transform(self):
        decomposer = ClusterDecomposer(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction,
            normalize_residuals=True,
            mu=0.8,
            verbose=True,
        )
        pipeline = ResidualDecomposer(
            decomposer=decomposer,
            n_steps=3,
            termination_criteria=["n_steps"],
            verbose=True,
        )

        residuals, transformed, assignments = pipeline.fit_transform(
            self.embeddings
        )

        residuals_t, transformed_t, _ = pipeline.transform(self.embeddings)

        print("res", residuals.shape)
        print("res_t", residuals_t.shape)

        print("tr", transformed.shape)
        print("tr_t", transformed_t.shape)

        print("res", residuals)
        print("res_t", residuals_t)

        print("tr", transformed)
        print("tr_t", transformed_t)

        self.assertTrue((residuals == residuals_t).all())
        self.assertTrue((transformed == transformed_t).all())


if __name__ == "__main__":
    unittest.main()
