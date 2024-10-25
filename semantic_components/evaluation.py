from scipy.sparse.csr import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn import metrics

import numpy as np


class OctisInterface:
    """
    An interface to the OCTIS library for topic modeling evaluation library.
    """

    def __init__(
        self,
        documents,
        representations,
        vocab,
        suffix="",
        report=print,
        verbose=True,
    ):
        """
        Initialize the interface with the documents and the representations.
        """
        self.representations = representations
        self.report = report
        self.vocab = vocab

        if "combined_id" in documents.columns:
            self.representations = self.representations[
                representations["component_id"]
                == representations["combined_id"]
            ]

        self.documents = documents.explode("component_id")
        self.suffix = suffix
        self.output = self.generate_octis_output(
            self.documents, self.representations
        )

        self.octis = None
        self.verbose = verbose

    def evaluate(self):
        """
        Evaluate the clustering results.
        """
        try:
            from octis.evaluation_metrics.diversity_metrics import TopicDiversity
            from octis.evaluation_metrics.coherence_metrics import Coherence
        except ImportError:
            if self.verbose:
                print()
                print("OCTIS not installed. Skipping this part of the evaluation.")
                print("If you still want to use this part of the evaluation, install OCTIS.")
                print("Or reinstall this package with the 'octis' option:")
                print(">>> pip install semantic-components[octis]")
                print()
            return

        # use octis to calculate coherence scores
        npmi = Coherence(
            texts=self.documents["tokenized"],
            topk=len(self.representations[0]),
            measure="c_npmi",
        )

        try:
            npmi_score = npmi.score(self.output)
        except Exception as e:
            if self.verbose:
                print("Exception in NPMI calculation.")
                print(e)
                print("Octis Output:", self.output)
            npmi_score = None

        self.report("OCTIS NPMI Coherence" + self.suffix + ": {}", npmi_score)

        cv = Coherence(
            texts=self.documents["tokenized"],
            topk=len(self.representations[0]),
            measure="c_v",
        )
        try:
            cv_score = cv.score(self.output)
        except Exception as e:
            if self.verbose:
                print("Exception in CV calculation.")
                print(e)
                print("Octis Output:", self.output)
            cv_score = None

        self.report("OCTIS CV Coherence" + self.suffix + ": {}", cv_score)

        # use octis to calculate topic diversity scores
        topic_diversity = TopicDiversity(topk=len(self.representations[0]))
        topic_diversity_score = topic_diversity.score(self.output)

        self.report(
            "OCTIS Topic Diversity" + self.suffix + ": {}",
            topic_diversity_score,
        )

    def generate_octis_output(self, documents, representations):
        """
        Generate an OCTIS output from the documents and the representations.
        """
        # generate octis output
        output = dict()
        output["topics"] = [
            [token for token in rep] for key, rep in representations.items()
        ]

        self.word2id = dict()
        for i, word in enumerate(self.vocab):
            self.word2id[word] = i
        self.id2word = {v: k for k, v in self.word2id.items()}

        output["topic-word-matrix"] = self.get_doc_word_matrix(
            output["topics"]
        )
        output["document-word-matrix"] = self.get_doc_word_matrix(
            documents["tokenized"]
        )

        d_topics = [id for id in documents["component_id"]]
        mx = max(d_topics)
        d_topics = [id if id != -1 else mx + 1 for id in d_topics]
        output["topic-document-matrix"] = self.get_topic_doc_matrix(d_topics)

        return output

    def get_doc_word_matrix(self, documents):
        """
        Get the document-word matrix.

        Parameters
        ----------
        documents : list
            A list of documents. Each document is a list of string-tokens.
        """
        data, row_ind, col_ind = [], [], []
        for i, doc in enumerate(documents):
            for word in doc:
                if word in self.word2id:
                    col_ind.append(self.word2id[word])
                    data.append(1)
                    row_ind.append(i)

        return csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(len(documents), len(self.word2id)),
        )

    def get_topic_doc_matrix(self, topics):
        """
        Get the topic-document matrix.

        Parameters
        ----------
        topics : list
            A list of topic-ids for each document.
        """
        data, row_ind, col_ind = [], [], []
        for i, component_id in enumerate(topics):
            col_ind.append(i)
            data.append(1)
            row_ind.append(component_id)

        return csr_matrix(
            (data, (row_ind, col_ind)),
            shape=(len(topics), len(self.documents)),
        )


class StepEvaluator:

    def __init__(
        self,
        metrics=[
            "emb_uniformity",
            "emb_alignment",
            "base_embedding_alignment",
            "topic_alignment_embedding",
            "embedding_norms",
            "matrix_embedding_norms",
            "octis_evaluation",
            "topic_noise",
            "overall_noise",
        ],
        report=print,
        verbose=True,
    ):
        self.metrics = metrics
        self.report = report
        self.verbose = verbose

    def evaluate(
        self, documents, embeddings, representations, vocab, hdbscan=None
    ):

        topic_ids = [id[0] for id in documents["component_id"]]

        self.report("")
        self.report("")
        self.report(
            "====================== CLUSTER EVALUATION RESULTS =========================="
        )
        if "emb_uniformity" in self.metrics:
            uniformity_euc, uniformity_cos = self.uniformity(
                embeddings, sample=10000
            )
            self.report(
                "Uniformity of embeddings (Euclidean): {}", uniformity_euc
            )
            self.report(
                "Uniformity of embeddings (Cosine): {}", uniformity_cos
            )
            self.report("")

        if (
            "base_embedding_alignment" in self.metrics
            and embeddings is not None
        ):
            alignment_euc, alignment_cos = self.alignment(
                embeddings, np.zeros(len(embeddings))
            )
            self.report(
                "Base-alignment of embeddings (Euclidean): {}", alignment_euc
            )
            self.report(
                "Base-alignment of embeddings (Cosine): {}", alignment_cos
            )
            self.report("")

        if (
            "topic_alignment_embedding" in self.metrics
            and embeddings is not None
        ):
            alignment_euc, alignment_cos = self.alignment(
                embeddings, np.array(topic_ids)
            )
            self.report(
                "Cluster-alignment of embeddings (Euclidean): {}",
                alignment_euc,
            )
            self.report(
                "Cluster-alignment of embeddings (Cosine): {}", alignment_cos
            )
            self.report("")

        if "embedding_norms" in self.metrics:
            l2_norms = np.linalg.norm(embeddings, ord=2, axis=1)

            self.report("L2-norm of embeddings: min: {}", [np.min(l2_norms)])
            self.report("L2-norm of embeddings: max: {}", [np.max(l2_norms)])
            self.report("L2-norm of embeddings: mean: {}", [np.mean(l2_norms)])
            self.report(
                "L2-norm of embeddings: median: {}", [np.median(l2_norms)]
            )
            self.report("Std of embedding L2-norms: {}", np.std(l2_norms))
            self.report("Sum of embedding L2-norms: {}", np.sum(l2_norms))
            self.report("")

            l1_norms = np.linalg.norm(embeddings, ord=1, axis=1)
            self.report("L1-norm of embeddings: min: {}", [np.min(l1_norms)])
            self.report("L1-norm of embeddings: max: {}", [np.max(l1_norms)])
            self.report("L1-norm of embeddings: mean: {}", [np.mean(l1_norms)])
            self.report(
                "L1-norm of embeddings: median: {}", [np.median(l1_norms)]
            )
            self.report("Std of embedding L1-norms: {}", np.std(l1_norms))
            self.report("Sum of embedding L1-norms: {}", np.sum(l1_norms))
            self.report("")

        if "matrix_embedding_norms" in self.metrics:
            self.report(
                "L2 Matrix norm of embeddings: {}",
                np.linalg.norm(embeddings, ord=2),
            )
            self.report(
                "L1 Matrix norm of embeddings: {}",
                np.linalg.norm(embeddings, ord=1),
            )

        if "topic_noise" in self.metrics:
            noiserate = self.get_noiserate(topic_ids)
            noise = self.get_total_noise(topic_ids)
            self.report(
                "Noiserate topics: {} (number of samples classified noise: {})",
                [noiserate, noise],
            )

        if "overall_noise" in self.metrics:
            max_ids = [max(ids) for ids in documents["component_id"]]
            noiserate = self.get_noiserate(max_ids)
            noise = self.get_total_noise(max_ids)
            self.report(
                "Noiserate overall: {} (number of samples classified noise: {})",
                [noiserate, noise],
            )

        if "octis_evaluation" in self.metrics:
            octis_interface = OctisInterface(
                documents,
                representations["representation"],
                vocab,
                report=self.report,
                verbose=self.verbose,
            )
            octis_interface.evaluate()
            # evaluate only first iteration/topic results
            documents_first = documents.copy()
            documents_first["component_id"] = [
                [id[0]] for id in documents["component_id"]
            ]
            octis_interface = OctisInterface(
                documents_first,
                representations[representations["iteration"] == 0][
                    "representation"
                ],
                vocab,
                suffix="_first",
                report=self.report,
                verbose=self.verbose,
            )
            octis_interface.evaluate()

        self.report(
            "======================================================================="
        )
        self.report("")
        self.report("")

    def token_overlap(self, component_representations, k=5):
        """
        Compute the token overlap of the components.
        """
        n_overlaps = 0
        n_overlapping = 0
        N = 0
        if (
            component_representations is None
            or len(component_representations) <= 1
        ):
            return 0, 0
        for i, c1 in enumerate(component_representations):
            if i >= len(component_representations) - 1:
                break
            for j, c2 in enumerate(component_representations[i + 1 :]):
                N += 1
                overlapping_toks = len(set(c1).intersection(set(c2)))
                n_overlapping += overlapping_toks

                if overlapping_toks >= k:
                    n_overlaps += 1

        return n_overlaps, n_overlapping / N

    def diversity(self, component_representations):
        """
        Compute the diversity of the components.
        """
        if component_representations is None:
            return 0
        unique_words = set()
        for component in component_representations:
            unique_words = unique_words.union(set(component))
        td = len(unique_words) / (10 * len(component_representations))
        return td

    def uniformity(self, embeddings, sample=-1):
        """
        Compute the uniformity of the embedding vectors.
        """

        if sample > 0:
            sample = min(sample, len(embeddings))
            embeddings = embeddings[
                np.random.choice(len(embeddings), sample, replace=False)
            ]

        # normalize embeddings
        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        # as in Xu et al. we use Euclidean distance, but also test cosine distance
        distances_euc = pairwise_distances(embeddings, metric="euclidean")
        distances_cos = pairwise_distances(embeddings, metric="cosine")

        # measure how well embeddings are uniformly distributed
        # separate the computations in the next line in separate lines following by a print of intermediate results
        # uniformity_euc = np.log(np.mean(np.exp(-2*np.square(distances_euc))))
        # uniformity_cos = np.log(np.mean(np.exp(-2*np.square(distances_cos))))
        squared = np.square(distances_euc)
        exp = np.exp(-2 * squared)
        mean = np.mean(exp)
        log = np.log(mean)
        uniformity_euc = log

        squared = np.square(distances_cos)
        exp = np.exp(-2 * squared)
        mean = np.mean(exp)
        log = np.log(mean)
        uniformity_cos = log

        # uniformity_cos = np.log(np.mean(np.exp(-2*np.square(distances_cos))))

        return uniformity_euc, uniformity_cos

    def alignment(self, embeddings, ground_truth, sample=10000):
        """
        Compute the alignment of the embedding vectors.
        """

        if ground_truth is None:
            raise ValueError("No ground truth provided.")

        embeddings = embeddings[ground_truth != -2]
        ground_truth = ground_truth[ground_truth != -2]

        if sample > 0:
            sample = min(sample, len(embeddings))
            ids = np.random.choice(
                list(range(len(embeddings))), sample, replace=False
            )
            embeddings = embeddings[ids]
            # print(ids)
            # print(len(ids))
            # print(ground_truth)
            ground_truth = ground_truth[ids]

        # normalize embeddings
        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        # as in Xu et al. we use Euclidean distance, but also test cosine distance
        distances_euc = pairwise_distances(embeddings, metric="euclidean")
        distances_cos = pairwise_distances(embeddings, metric="cosine")

        # compute a matrix with pairwise equality checks, 1 if equal, 0 if not
        alignment_matrix = (
            np.tile(ground_truth, (len(ground_truth), 1))
            == np.tile(ground_truth, (len(ground_truth), 1)).transpose()
        )

        # measure how well embeddings are uniformly distributed
        alignment_euc = np.mean(np.square(distances_euc)[alignment_matrix])
        alignment_cos = np.mean(np.square(distances_cos)[alignment_matrix])

        return alignment_euc, alignment_cos

    def get_cluster_purity(self, ids, ground_truth, sample=10000):
        """
        Compute the cluster purity of the clustering.

        Taken from:
        https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
        """
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(
            ground_truth, ids
        )
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(
            contingency_matrix
        )

    def get_noiserate(self, ids):
        """
        Compute the noiserate of the clustering.
        """
        if type(ids) is list:
            ids = np.array(ids)
        return np.sum(ids == -1) / len(ids)

    def get_total_noise(self, ids):
        """
        Compute the number of samples classified as noise.
        """
        if type(ids) is list:
            ids = np.array(ids)
        return np.sum(ids == -1)


class OverlapEvaluator:

    def __init__(
        self,
        metrics=["token_overlap", "diversity"],
        report=print,
        verbose=True,
    ):
        self.metrics = metrics
        self.report = report
        self.verbose = verbose

    def evaluate(self, representations):

        if "token_overlap" in self.metrics:
            n_overlaps, overlap = self.token_overlap(representations)
            self.report(
                "Token overlap: {} (average number of overlapping tokens: {})",
                [n_overlaps, overlap],
            )

        self.report(
            "======================================================================="
        )
        self.report("")
        self.report("")

    def token_overlap(self, representations, k=5):
        """
        Compute the token overlap of the components.
        """
        n_overlaps = 0
        n_overlapping = 0
        N = 0
        if representations is None or len(representations) <= 1:
            return 0, 0
        for i, c1 in enumerate(representations):
            if i >= len(representations) - 1:
                break
            for j, c2 in enumerate(representations[i + 1 :]):
                N += 1
                overlapping_toks = len(set(c1).intersection(set(c2)))
                n_overlapping += overlapping_toks

                if overlapping_toks >= k:
                    n_overlaps += 1

        return n_overlaps, n_overlapping / N
