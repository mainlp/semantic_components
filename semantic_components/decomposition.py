from numpy.linalg import norm
from sklearn.preprocessing import normalize
from hdbscan import HDBSCAN
from umap import UMAP
import numpy as np
from semantic_components.cluster_pipeline import ClusterPipeline
from copy import deepcopy


class Decomposer:
    """
    Decompose embeddings into semantic components.

    Takes a set of embeddings and returns a set of components, assignments
    and residuals.
    """

    def __init__(self):
        pass

    def fit(self, embeddings):
        """
        Fit the decomposer.
        """
        pass

    def transform(self, embeddings):
        """
        Transform the embeddings.
        """
        pass

    def fit_transform(self, embeddings):
        """
        Fit and transform the embeddings.
        """
        pass

    def set_params(self, new):
        raise NotImplementedError("set_params hasn't been implemented yet.")


class ResidualDecomposer(Decomposer):
    """
    A decomposer that keeps re-applying another decomposer on the residuals of
    the previous decomposition step.
    """

    def __init__(
        self,
        decomposer=None,
        name="residual_step",
        verbose=False,
        n_steps=10,
        termination_criteria=None,
        parameter_schedule=None,
        keep=None,
        eps_component=0.01,
        alpha_decomposition=0.01,
        n_history=5,
        min_new_components=5,
        log=None,
        report=print,
    ):
        """
        Initialize the ResidualDecomposer.
        """
        self.decomposer = decomposer
        self.name = name
        self.verbose = verbose
        self.n_steps = n_steps
        self.termination_crit = termination_criteria
        self.parameter_schedule = parameter_schedule
        self.keep = keep
        self.report = report
        self.eps_component = eps_component
        self.alpha_decomposition = alpha_decomposition
        self.n_comp_history = []
        self.n_history = n_history
        self.min_new_components = min_new_components
        self.log = lambda x: log(x) if log is not None else None
        self.components = []
        self.decomposers = []

    def fit(self, embeddings):
        """
        Fit the decomposer.
        """

        residuals = embeddings
        assignments = []
        new_assignments = []
        step = 0
        scores = []
        offset = 0

        while not self.termination_criterion(residuals, new_assignments, step):

            self.decomposer = self.get_next_decomposer(self.decomposer)

            if self.verbose:
                self.report("")
                self.report(f"ResidualDecomposer: Step {step}.")
                self.report("=====================================")
                self.report(
                    f"ResidualDecomposer: Applying decomposer {self.decomposer.name}."
                )
                self.report(
                    f"ResidualDecomposer_step={step}: Current number of components: {offset}."
                )
                # self.report(f"ResidualDecomposer_step={step}: Current residual norm: {norm(residuals, ord=2)}.")
                self.report("")

            residuals, transformed, new_assignments = (
                self.decomposer.fit_transform(residuals)
            )

            has_noise = -1 in new_assignments

            new_assignments = [
                a + offset if a > -1 else -1 for a in new_assignments
            ]

            assignments.append(new_assignments)
            scores.append(transformed)

            self.n_comp_history.append(
                len(np.unique(new_assignments)) - 1 * has_noise
            )

            self.decomposers.append(self.decomposer)

            step += 1
            offset += (
                len(np.unique(new_assignments)) - 1 * has_noise
            )  # do not count noise cluster
            self.components += sorted(list(np.unique(new_assignments)))

        # make it a list per sample and not a list per run
        assignments = [
            [a[i] for a in assignments] for i in range(len(assignments[0]))
        ]

        return residuals, np.concatenate(scores, axis=1), assignments

    def fit_transform(self, embeddings):
        return self.fit(embeddings)

    def transform(self, embeddings):

        all_scores = []
        assignments = []

        residual = embeddings

        for decomposer in self.decomposers:
            residual, scores, new_assignments = decomposer.transform(residual)

            if scores is not None:
                all_scores.append(scores)
                assignments.append(new_assignments)

        return residual, np.concatenate(all_scores, axis=1), assignments

    def get_next_decomposer(self, decomposer):
        """
        Get the next decomposer in the schedule.
        """
        next = decomposer.get_unfitted_copy()

        if self.parameter_schedule is not None:
            next.set_params(self.parameter_schedule[0])
            self.parameter_schedule = self.parameter_schedule[1:]

        return next

    def termination_criterion(self, embeddings, ids, i):
        """
        Check if the termination criterion is met.
        """
        terminated = False
        if type(self.termination_crit) not in [list, tuple, set]:
            self.termination_crit = [self.termination_crit]

        if i <= 0:
            return False

        for crit in self.termination_crit:
            if crit == "residual_norm":
                if self.verbose:
                    print("Termination criterion: Checking residual norm.")
                if norm(embeddings, ord=2) < self.eps_component:
                    terminated = True
            elif crit == "new_components" and i > 1:
                if self.verbose:
                    print(
                        "Termination criterion: Checking number of new components."
                    )
                if len(set(ids)) < self.min_new_components:
                    terminated = True
            elif crit == "new_components_history":
                # self.n_comp_history.append(len(set(ids)))
                if self.verbose:
                    print(
                        "Termination criterion: Checking history of new components."
                    )
                    print("History: ", self.n_comp_history)
                    print("Min new components: ", self.min_new_components)
                recent_history = self.n_comp_history[
                    -min(len(self.n_comp_history), self.n_history) :
                ]
                if len(recent_history) == self.n_history:
                    if all(
                        [n < self.min_new_components for n in recent_history]
                    ):
                        terminated = True
            elif crit == "max_iterations" or crit == "n_steps":
                if self.verbose:
                    print(
                        "Termination criterion: Checking number of iterations."
                    )
                if i >= self.n_steps:
                    terminated = True

        return terminated

    def get_component_repr(self, i):
        """
        Get the representation of a component.
        """
        iteration = 0
        offset = 0
        for j, n in enumerate(self.n_comp_history):
            if offset + n > i:
                iteration = j
                break
            offset += n

        return self.decomposers[iteration].get_component_repr(i - offset)


class ClusterDecomposer(Decomposer):
    """
    Decompose embeddings into semantic components using the clustering
    approach.
    """

    def __init__(
        self,
        cluster_algorithm=None,
        dim_reduction=None,
        normalize_residuals=False,
        min_cluster_size=150,
        min_samples=50,
        n_neighbours=20,
        umap_n_components=5,
        n_epochs=200,
        min_dist=0.0,
        eps_component=0.01,
        alpha_decomposition=0.01,
        mu=1.0,
        name="cluster_step",
        stopwords_path=None,
        centroid_normalization="l2",
        verbose=False,
        log=None,
        report=print,
    ):
        """
        Initialize the ClusterDecomposer.
        """
        self.cluster_algorithm = cluster_algorithm
        self.dim_reduction = dim_reduction
        self.min_cluster_size = min_cluster_size
        self.normalize_residuals = normalize_residuals
        self.min_samples = min_samples
        self.n_neighbours = n_neighbours
        self.umap_n_components = umap_n_components
        self.n_epochs = n_epochs
        self.mu = mu
        self.min_dist = min_dist
        self.name = name
        self.eps_component = eps_component
        self.alpha_decomposition = alpha_decomposition
        self.centroid_normalization = centroid_normalization
        self.verbose = verbose
        self.stopwords_path = stopwords_path
        self.log = lambda x, y: log(x, y) if log is not None else None
        self.report = report
        self.component_vectors = []
        self.component_matrix = None

        if self.cluster_algorithm is None:
            self.cluster_algorithm = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
                min_samples=min_samples,
            )

        if self.dim_reduction is None:
            self.dim_reduction = UMAP(
                n_neighbors=n_neighbours,
                n_components=umap_n_components,
                min_dist=min_dist,
                n_epochs=n_epochs,
                metric="cosine",
                random_state=42,
            )

        self.cluster_pipeline = ClusterPipeline(
            deepcopy(self.cluster_algorithm), 
            deepcopy(self.dim_reduction),
            verbose=self.verbose,
            report=self.report,
            log=self.log,
            name=self.name,
        )

    def get_unfitted_copy(self):

        return ClusterDecomposer(
            cluster_algorithm=deepcopy(self.cluster_algorithm),
            dim_reduction=deepcopy(self.dim_reduction),
            normalize_residuals=self.normalize_residuals,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            n_neighbours=self.n_neighbours,
            umap_n_components=self.umap_n_components,
            n_epochs=self.n_epochs,
            min_dist=self.min_dist,
            eps_component=self.eps_component,
            alpha_decomposition=self.alpha_decomposition,
            mu=self.mu,
            name=self.name,
            stopwords_path=self.stopwords_path,
            centroid_normalization=self.centroid_normalization,
            verbose=self.verbose,
            log=self.log,
            report=self.report,
        )

    def fit(self, embeddings):
        """
        Fit the decomposer.
        """
        if self.cluster_pipeline is None:
            self.cluster_pipeline = ClusterPipeline(
                self.cluster_algorithm,
                dim_reduction=self.dim_reduction,
                store_results=self.store_results,
                name=self.name,
                verbose=self.verbose,
                report=self.report,
                log=self.log,
            )

        ids, probs = self.cluster_pipeline.fit_transform(embeddings)

        residuals, scores, ids = self.decompose(embeddings, ids)

        self.component_matrix = np.array(self.component_vectors).T

        return residuals, scores, ids

    def transform(self, embeddings, cluster=False):
        """
        Transform the embeddings.
        """

        if self.component_matrix is None:
            print(
                "ClusterDecomposer: has not been fit to data yet. Fitting..."
            )
            return self.fit(embeddings)

        if self.component_matrix.shape[0] == 0:
            print("ClusterDecomposer: Empty component matrix.")
            return embeddings, None, None

        if (
            self.mu == 1.0
            and not self.normalize_residuals
            and self.centroid_normalization == "l2"
            and self.alpha_decomposition == 0
        ):
            # linear decomposition, can just use dot product (more efficient)
            transformed = embeddings.dot(self.component_matrix)
            residuals = embeddings - transformed.dot(self.component_matrix.T)

        else:
            # we need to iteratively go over the components and decompose
            residuals = embeddings
            transformed = []
            for component in self.component_vectors:
                residuals, scores = self.decomposition_step(
                    residuals, component
                )
                transformed.append(scores)
            transformed = np.array(transformed).T

        return residuals, transformed, None

    def fit_transform(self, embeddings):
        """
        Fit and transform the embeddings.
        """
        return self.fit(embeddings)

    def compute_component(self, embeddings):
        """
        Compute the centroid of a set of embeddings. Behaviour depends
        on the following class parameters:
        - centroid_normalization: l2, None, any valid sklearn norm argument
        """

        centroid = np.mean(embeddings, axis=0)

        if self.centroid_normalization is not None:
            centroid = normalize(
                centroid.reshape(1, -1),
                norm=self.centroid_normalization,
                axis=1,
            ).reshape(-1)

        return centroid

    def decompose(self, embeddings, ids):
        """
        Decompose the embeddings into semantic components.
        Returns residual embeddings.
        """

        component_scores = []

        no_of_concepts = max(ids) + 1

        if self.verbose:
            print("Decomposing {} concepts.".format(no_of_concepts))

        for concept_id in range(no_of_concepts):
            # compute centroid
            relevant_embeddings = embeddings[ids == concept_id]

            component = self.compute_component(relevant_embeddings)

            if norm(component, ord=2) > self.eps_component:
                embeddings, scores = self.decomposition_step(
                    embeddings, component
                )
                component_scores.append(scores)
                self.component_vectors.append(component)

            else:
                # this component is too small to be considered
                # switch concept_id to -1 in ids
                ids[ids == concept_id] = -1
                if self.verbose:
                    print(
                        "Component {}'s norm ({}) is smaller than eps={}\
                           and thus not considered.".format(
                            concept_id,
                            norm(component, ord=2),
                            self.eps_component,
                        )
                    )

        return (
            embeddings,
            np.array(component_scores).reshape((embeddings.shape[0], -1)),
            ids,
        )

    def decomposition_step(self, embeddings, component):
        """
        Perform one decomposition step.

        Dependent on the following class parameters:
        - decomposition: full, mu, unnormalized
        - normalize_residuals: True, False
        - mu: float
        """
        factors = embeddings.dot(component)

        if self.mu != 1.0:
            factors = factors * self.mu

        if self.alpha_decomposition != 0.0:
            sims = factors / (
                norm(component, ord=2) * norm(embeddings, ord=2, axis=1)
            )
            # set small values to zero
            factors = factors * (sims >= self.alpha_decomposition)

        centroid_stack = np.tile(component, (factors.shape[0], 1))
        linear_subcomponents = np.multiply(
            centroid_stack, np.reshape(factors, (len(factors), 1))
        )
        embeddings = np.subtract(embeddings, linear_subcomponents)

        if self.normalize_residuals:
            embeddings = normalize(embeddings, norm="l2", axis=1)

        return embeddings, factors

    def get_component_repr(self, i):
        """
        Get the representation of a component.
        """
        if not self.component_vectors:
            print("ClusterDecomposer: No components have been fitted.")
            return None
        
        if i >= len(self.component_vectors):
            print("ClusterDecomposer: Component index out of bounds.")
            return None
        
        return self.component_vectors[i]


class AEDecomposer(Decomposer):
    pass
