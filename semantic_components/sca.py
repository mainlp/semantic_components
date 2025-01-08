from semantic_components.representation import CTFIDFRepresenter, MedoidRepresenter
import semantic_components.evaluation
from semantic_components.decomposition import ClusterDecomposer, ResidualDecomposer
import pandas as pd
import os
import pickle
import re
import time


class SCA:
    """
    Semantic component analysis (SCA).

    Parameters
    ----------

    cluster_algorithm : optional
        The clustering algorithm to use. Default is None, which will use the standard HDBSCAN algorithm with the 
        specified parameters.

    dim_reduction_algorithm : optional
        The dimensionality reduction algorithm to use. Default is None, which will use the standard UMAP algorithm as 
        described in the paper.

    normalize_components : bool, optional
        Whether to normalize the components. Default is True.

    termination_crit : list, optional
        The termination criteria for the decomposition. 
        Default is ["max_iterations", "residual_norm", "new_components_history"].

    eps_component : float, optional
        A parameter ensuring computational stability in case normalize_components is set to False. Only components whose
          centroid is higher than eps_component will be considered. Default is 0.01.

    alpha_decomposition : float, optional
        The decomposition threshold parameter. Higher values lead to sparser component distributions and allows for 
        superposition of components. Should be in < 1.0. Default is 0.0.

    mu : float, optional
        The decomposition strength parameter. Values should be chosen in [0, 1]. Default is 1.0.

    min_tokens : int, optional
        The minimum number of tokens in a document. Documents with less tokens are not considered. Default is 3.

    hdbscan_min_cluster_size : int, optional
        The minimum cluster size for the HDBSCAN algorithm. Default is 100.

    hdbscan_min_samples : int, optional
        The minimum number of samples for the HDBSCAN algorithm. Default is 50.

    umap_n_neighbours : int, optional
        The number of neighbours for the UMAP algorithm. Default is 20.

    umap_n_epochs : int, optional
        The number of epochs for the UMAP algorithm. Default is 200.

    umap_n_components : int, optional
        The number of components for the UMAP algorithm. Default is 5.

    umap_min_dist : float, optional
        The minimum distance for the UMAP algorithm. Default is 0.0.

    new_components : int, optional
        If the number of new components falls below this threshold for n_history iterations, the procedure terminates. 
        Default is 5.

    combine_overlap_threshold : float, optional
        The threshold for combining components based on overlap. Default is 0.5.

    n_grams : int, optional
        The number of tokens to consider for the c-TF-IDF representation. Default is 1.

    n_history : int, optional
        The number of iterations to consider for terminating the procedure based on new_components. Default is 2.

    max_iterations : int, optional
        The maximum number of iterations to run. Default is 10.

    language : str, optional
        The language. Default is "english". For the languages we discuss in the paper, this influences the tokenizer and
          the stopword list for the representation.
    
    tokenizer : optional
        Pass a custom tokenizer. The tokenizer should implement a `tokenize` and `__call__` function. See
        `semantic_components.representation.GenericTokenizer` for a minimal example. Default is None.

    evaluation : bool, optional
        Whether to evaluate the model. Default is False.

    stopwords_path : str, optional
        The path to the stopwords file. Default is None, which will use the standard stopwords for the specified 
        language.

    verbose : bool, optional
        Whether to print the progress. Default is False.

    logging : bool, optional
        Whether to log the progress. Default is False.

    store_results : bool, optional
        Whether to store the results. Default is False.

    path : str, optional
        The path to store the results. Default is None.
    """

    def __init__(
        self,
        cluster_algorithm=None,
        dim_reduction_algorithm=None,
        normalize_components=True,
        termination_crit=[
            "max_iterations",
            "residual_norm",
            "new_components_history",
        ],
        eps_component=0.01,
        alpha_decomposition=0.0,
        mu=1.0,
        min_tokens=3,
        hdbscan_min_cluster_size=100,
        hdbscan_min_samples=50,
        umap_n_neighbours=20,
        umap_n_epochs=200,
        umap_n_components=5,
        umap_min_dist=0.0,
        new_components=5,
        combine_overlap_threshold=0.5,
        n_grams=1,
        n_history=2,
        max_iterations=10,
        language="english",
        tokenizer=None,
        evaluation=False,
        stopwords_path=None,
        verbose=False,
        logging=False,
        store_results=False,
        path="results/",
    ):
        self.cluster_algorithm = cluster_algorithm
        self.dim_reduction_algorithm = dim_reduction_algorithm
        self.normalize_components = normalize_components
        self.termination_crit = termination_crit
        self.eps_component = eps_component
        self.alpha_decomposition = alpha_decomposition
        self.mu = mu
        self.min_tokens = min_tokens
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.umap_n_neighbours = umap_n_neighbours
        self.umap_n_epochs = umap_n_epochs
        self.umap_n_components = umap_n_components
        self.umap_min_dist = umap_min_dist
        self.new_components = new_components
        self.n_history = n_history
        self.max_iterations = max_iterations
        self.language = language
        self.tokenizer = tokenizer
        self.stopwords_path = stopwords_path
        self.verbose = verbose
        self.logging = logging
        self.store_results = store_results
        self.path = path
        self.reports = []
        self.iteration = []
        self.n_grams = n_grams
        self.representations = None
        self.evaluation = evaluation
        self.logs = dict()
        self.combine_overlap_threshold = combine_overlap_threshold
        self.component_mapping = None

    def fit(self, documents, embeddings):
        """
        Fit the SCA model.

        Parameters
        ----------
        documents : pd.DataFrame
            The documents to fit the model on.

        embeddings : np.array
            The embeddings to fit the model on.
        """
        residuals, scores, ids = self.decompose(documents, embeddings)

        if documents is None:
            print("No documents provided. Cannot represent.")
            return scores, residuals, ids

        if "text_preprocessed" not in documents.columns:
            print(
                "No preprocessed text provided. Adding preprocessed column in documents."
            )
            documents["text_preprocessed"] = documents["text"]

        documents, self.representations = self.represent(
            documents, embeddings, ids
        )

        if self.evaluation:
            self.evaluate(documents, embeddings, self.representations, ids)

        if self.store_results:
            self.save(self.path)

        return scores, residuals, ids

    def fit_transform(self, documents, embeddings):
        """
        Fit the SCA model and transform the data.
        """
        ids, scores, residuals = self.fit(documents, embeddings)
        return scores, residuals, ids

    def transform(self, embeddings):
        """
        Transform the data.
        """
        residuals, scores, assignments = self.decomposer.transform(embeddings)
        return scores, residuals, assignments

    def decompose(self, documents, embeddings):
        """
        Decompose the data.
        """
        sub_decomposer = ClusterDecomposer(
            cluster_algorithm=self.cluster_algorithm,
            dim_reduction=self.dim_reduction_algorithm,
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            n_neighbours=self.umap_n_neighbours,
            n_epochs=self.umap_n_epochs,
            umap_n_components=self.umap_n_components,
            min_dist=self.umap_min_dist,
            eps_component=self.eps_component,
            alpha_decomposition=self.alpha_decomposition,
            mu=self.mu,
            verbose=self.verbose,
            log=self.log,
            report=self.report,
        )

        self.decomposer = ResidualDecomposer(
            decomposer=sub_decomposer,
            n_steps=self.max_iterations,
            termination_criteria=self.termination_crit,
            verbose=self.verbose,
            log=self.log,
            report=self.report,
            n_history=self.n_history,
            eps_component=self.eps_component,
            alpha_decomposition=self.alpha_decomposition,
            min_new_components=self.new_components,
        )

        residuals, scores, ids = self.decomposer.fit_transform(embeddings)

        documents["component_id"] = ids
        self.components = list(range(-1, sum(self.decomposer.n_comp_history)))
        self.iteration = [-1] + sum(
            [[i] * n for i, n in enumerate(self.decomposer.n_comp_history)], []
        )

        return residuals, scores, ids

    def represent(self, documents, embeddings, assignments):
        """
        Represent the data.
        """
        representations = pd.DataFrame(
            {"id": self.components, "iteration": self.iteration}
        )
        representations = representations.set_index("id")

        ctfidf_representer = CTFIDFRepresenter(
            language=self.language,
            tokenizer=self.tokenizer,
            stopwords_path=self.stopwords_path,
            verbose=self.verbose,
            log=self.log,
            report=self.report,
            n_grams=self.n_grams,
        )

        ctfidf_representer.fit(documents)
        representations_ctfidf = ctfidf_representer.transform(documents)
        representations_ctfidf = representations_ctfidf.rename(
            columns={"component_id": "id", "iteration": "iteration"}
        )

        representations_ctfidf = representations_ctfidf.set_index("id")
        representations = representations.join(
            representations_ctfidf, on="id", how="left", rsuffix="_ctfidf"
        )

        documents["tokenized"] = ctfidf_representer.tokenized
        self.vocab = ctfidf_representer.vocab

        if self.combine_overlap_threshold > 0:
            self.component_mapping = self.combine_ids_by_overlap(
                representations
            )
            representations["combined_id"] = representations.index.map(
                self.component_mapping
            )
            documents["component_id"] = documents["component_id"].apply(
                lambda x: [self.component_mapping[e] for e in x]
            )

            self.report(
                "Number of components after merging: {}",
                len(representations["combined_id"].unique()),
            )

        # medoids
        medoid_representer = MedoidRepresenter()
        result = medoid_representer.fit(documents, embeddings, self.decomposer)
        result = result.set_index("id")
        representations = representations.join(
            result, on="id", how="left", rsuffix="_medoid"
        )

        return documents, representations

    def combine_ids_by_overlap(self, representations):
        """
        Calculate the overlap.
        """
        swaps = []
        for i in representations.index:
            for j in representations.index:
                merged = False
                if i > j and i != -1 and j != -1 and not merged:
                    
                    # apparently, this can happen if you have many small samples
                    if (representations.loc[i]['representation'] is None 
                        or representations.loc[j]['representation'] is None):
                        self.report(f"combine_ids_by_overlap: Skipping {i} and {j} because of None representation")
                        continue

                    if (type(representations.loc[i]['representation']) is not list 
                        or type(representations.loc[j]['representation']) is not list):
                        self.report(f"combine_ids_by_overlap: Skipping {i} and {j} because of non-list representation")
                        continue

                    if (len(representations.loc[i]['representation']) == 0 or
                         len(representations.loc[j]['representation']) == 0):
                        self.report(f"combine_ids_by_overlap: Skipping {i} and {j} because of empty representation")
                        continue

                    overlap = (
                        len(
                            set(
                                representations.loc[i]["representation"]
                            ).intersection(
                                representations.loc[j]["representation"]
                            )
                        )
                        / 10
                    )
                    if overlap > self.combine_overlap_threshold:
                        swaps.append((i, j))
                        merged = True

        new_ids = dict([(i, i) for i in range(-1, len(representations) - 1)])
        for i, j in swaps:
            new_ids[i] = j
        return new_ids

    def evaluate(self, documents, embeddings, representations, ids):
        """
        Evaluate the SCA model.
        """
        evaluator = semantic_components.evaluation.StepEvaluator(report=self.report)
        evaluator.evaluate(documents, embeddings, representations, self.vocab)

    def save(self, path):
        """
        Save the results of the SCA model.
        """
        self.store_reports(path)
        self.store_logs(path)

    def report(self, report, values=None, prefix="", log=True, max_len=1024):
        """
        Adds a report to the list of reports and prints it to the console.

        Args:
            report (str): The report to add.
            prefix (str, optional): A prefix to add to the report. Defaults to "".
        """

        if self.verbose:
            if log and values is not None:
                self.logs["'" + str(report) + "'@" + str(time.time())] = values

            message = "> "
            if prefix:
                message += str(prefix) + ": "

            if values is not None:
                if type(values) is not list:
                    values = [values]
                message += str(report).format(*values)
            else:
                message += str(report)

            print(message[: min(len(message), max_len)])

            self.reports.append(message)

    def log(self, log_key, log_value, existing="overwrite", sep="\n"):
        """
        Log a value.
        """
        if log_key in self.logs.keys():
            if existing == "append":
                self.logs[log_key] = self.logs[log_key] + sep + log_value
            elif existing == "overwrite":
                self.logs[log_key] = log_value
        else:
            self.logs[log_key] = log_value

    def store_reports(self, results_directory):
        """
        Store the results to a file.
        """
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        if self.reports is not None:
            reports_path = (
                results_directory
                + "/reports@("
                + str(time.asctime())
                + ").txt"
            )
            with open(reports_path, "w") as f:
                f.writelines([line + "\n" for line in self.reports])

    def store_logs(self, results_directory):
        """
        Store the logs to a file.
        """
        logs_path = (
            results_directory + "/logs@(" + str(time.asctime()) + ").pkl"
        )
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        with open(logs_path, "wb") as f:
            pickle.dump(self.logs, f)

    def store_embeddings(self, path):
        """
        Store the embeddings to a file.
        """
        if not os.path.exists(re.sub(r"\/.*$", "", path)):
            os.makedirs(re.sub(r"\/.*$", "", path))
        with open(path, "wb") as fp:
            pickle.dump(self.embeddings, fp)

    def get_representation_string(self, ids=None):
        """
        Get the representations.
        """
        representation_str = ""
        for i, component in enumerate(self.components):
            if ids is None or i in ids:
                representation_str += f"Component {component}:\n"
                for rep in self.representations.columns:
                    if rep not in []:
                        representation_str += (
                            f"  {rep}: {self.representations[rep].iloc[i]}\n"
                        )
        return representation_str

    def get_scores(self, embeddings, top_n=-1):
        """
        Get the scores.
        """
        residuals, scores, assignments = self.decomposer.transform(embeddings)

        if top_n >= 0:
            scores = scores[:, : min(top_n, scores.shape[1])]

        return scores, residuals
