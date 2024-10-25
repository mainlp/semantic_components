"""
Representation of the components found. 

The code of the c-TF-IDF representer has been largely adapted from the original BERTopic implementation by 
Maarten Grootendorst released under MIT license. 
Full license under: https://github.com/MaartenGr/BERTopic/blob/master/LICENSE
Code: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/vectorizers/_ctfidf.py
"""

from typing import List
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.utils import check_array
import numpy as np
import scipy.sparse as sp
import pandas as pd
import re
import collections
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, Mapping
from scipy.sparse import csr_matrix
from numpy.linalg import norm

import jieba


class MedoidRepresenter:

    def __init__(
        self,
    ):
        self.medoids = []
        self.id = []

    def fit(self, documents, embeddings, decomposer):
        """
        Fit the medoid model to the documents.
        """
        self.medoids = []

        for i in np.unique(decomposer.components):
            centroid = decomposer.get_component_repr(i)
            self.id.append(i)
            self.medoids.append(
                self.get_medoid(documents, embeddings, centroid)
            )

        return pd.DataFrame({"id": self.id, "representation": self.medoids})

    def transform(self, documents, embeddings, decomposer):
        return pd.DataFrame({"id": self.id, "representation": self.medoids})

    def fit_transform(self, documents, embeddings, decomposer):
        return self.fit(documents, embeddings, decomposer)

    def get_medoid(self, documents, embeddings, centroid):
        """
        Get the medoid of the documents.
        """
        medoid = []
        # compute cosine of each document with centroid
        sims = np.dot(embeddings, centroid) / (
            norm(embeddings, axis=1) * norm(centroid)
        )
        argmax = np.argmax(sims)
        medoid = documents.iloc[argmax]["text"]
        return medoid


class CTFIDFRepresenter:

    def __init__(
        self,
        tokenizer=None,
        top_n_words=10,
        language="en",
        min_tokens=3,
        n_grams=1,
        stopwords=None,
        stopwords_path=None,
        verbose=False,
        log=None,
        report=print,
    ):
        """
        Initialize the CTF-IDF model.
        """
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.vocab = None
        self.stopwords = stopwords
        self.stopwords_path = stopwords_path
        self.report = report
        self.language = language
        self.top_n_words = top_n_words
        self.verbose = verbose
        self.log = lambda x: None if log is None else log(x)
        self.n_grams = n_grams

        if self.stopwords is None:
            self.stopwords = self.load_stopwords(stopwords_path)

        if self.tokenizer is None:
            if self.language in {"zh", "ch", "chinese", "Chinese"}:
                self.token_pattern = r"(?u)\b\w\w+\b"
                self.tokenizer = ChineseTokenizer(stopwords=self.stopwords)
            elif self.language in {"en", "eng", "english", "English"}:
                self.token_pattern = r"(?u)\b\w\w+\b"
                self.tokenizer = EnglishTokenizer(stopwords=self.stopwords)
            else:
                self.token_pattern = r"\b..+\b"
                self.tokenizer = GenericTokenizer(stopwords=self.stopwords)

        self.ctfidf_model = None

    def fit(self, documents):
        """
        Fit the CTF-IDF model to the documents.
        """
        # get vocab
        self.vocab, self.tokenized = self.get_vocab(
            documents["text_preprocessed"]
        )

        # as each document can have multiple component ids, we need to flatten the data
        # to have a copy of each document for each component id
        expl_documents = documents.explode("component_id")

        # fit the c-tf-idf model
        self.ctfidf_model = CTFIDFInterface(
            top_n_words=self.top_n_words,
            n_grams=self.n_grams,
            vocabulary=self.vocab,
            stopwords=self.stopwords,
            tokenizer=self.tokenizer,
            token_pattern=self.token_pattern,
        )
        self.ctfidf_model.fit(expl_documents)

    def transform(self, documents):
        """
        Transform the documents into CTF-IDF representations.
        """
        return self.ctfidf_model.get_topic_info()

    def fit_transform(self, documents):
        """
        Fit the CTF-IDF model to the documents and transform them into CTF-IDF representations.
        """
        self.fit(documents)
        return self.transform(documents)

    def get_vocab(self, documents):
        """
        Get the vocabulary of the documents.
        """
        vocab = collections.Counter()

        tokenized = []

        for doc in documents:
            tokens = self.tokenizer(doc)
            vocab.update(tokens)
            tokenized.append(tokens)

        vocab = [
            word
            for word, frequency in vocab.items()
            if frequency >= self.min_tokens
        ]

        documents.tokenized = tokenized

        self.vocab = vocab

        return vocab, tokenized

    def load_stopwords(self, path):
        """
        Load stopwords from a file.
        """
        stopwords = []
        if path is None:
            self.report("No stopwords path provided.")
            return stopwords

        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                stopwords.append(line.strip("\n").strip())
        self.stopwords = stopwords
        return stopwords


class CTFIDFInterface:
    """
    A class to compute the C-TF-IDF weighting of a clustering result.
    """

    def __init__(
        self,
        token_pattern=r"(?u)\b\w\w+\b",
        tokenizer=None,
        top_n_words=10,
        n_grams=1,
        stopwords=[],
        vocabulary=None,
    ):
        """
        Initialize the CTF-IDF model.
        """
        self.vectorizer_model = CountVectorizer(
            ngram_range=(1, n_grams),
            stop_words=stopwords,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            token_pattern=token_pattern,
        )
        self.top_n_words = top_n_words
        self.ctfidf_model = ClassTfidfTransformer()

        self.n_grams = n_grams

        self.topic_sizes_ = None
        self.topic_representations_ = None
        self.topic_labels_ = None
        self.c_tf_idf_ = None

    def fit(self, documents):
        """
        Fit the CTF-IDF model to the documents.
        """
        self._extract_topics(documents)

        if self.topic_sizes_ is None:
            self._update_topic_size(documents)

    def _preprocess_text(self, documents):
        """Basic preprocessing of text

        Steps:
            * Replace \n and \t with whitespace
            Removed: * Only keep alpha-numerical characters
        """
        cleaned_documents = [doc.replace("\n", " ") for doc in documents]
        cleaned_documents = [
            doc.replace("\t", " ") for doc in cleaned_documents
        ]
        cleaned_documents = [
            re.sub(r"[^\u4e00-\u9fffA-Za-z0-9 ]+", "", doc)
            for doc in cleaned_documents
        ]
        return cleaned_documents

    def _extract_topics(
        self,
        documents: pd.DataFrame,
        embeddings: np.ndarray = None,
        mappings=None,
        verbose: bool = False,
    ):
        """Extract topics from the clusters using a class-based TF-IDF

        Taken from: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py

        Arguments:
            documents: Dataframe with documents and their corresponding IDs
            embeddings: The document embeddings
            mappings: The mappings from topic to word
            verbose: Whether to log the process of extracting topics

        Returns:
            c_tf_idf: The resulting matrix giving a value (importance score) for each word per topic
        """
        documents_per_topic = documents.groupby(
            ["component_id"], as_index=False
        ).agg({"text": " ".join})
        self.c_tf_idf_, words = self._c_tf_idf(documents_per_topic)
        self.topic_representations_ = self._extract_words_per_topic(
            words, documents
        )
        self.topic_labels_ = {
            key: f"{key}_" + "_".join([word[0] for word in values[:4]])
            for key, values in self.topic_representations_.items()
        }

    def _c_tf_idf(
        self,
        documents_per_topic: pd.DataFrame,
        fit: bool = True,
        partial_fit: bool = False,
    ):
        """Calculate a class-based TF-IDF where m is the number of total documents.

        Taken from: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py

        Arguments:
            documents_per_topic: The joined documents per topic such that each topic has a single
                                 string made out of multiple documents
            m: The total number of documents (unjoined)
            fit: Whether to fit a new vectorizer or use the fitted self.vectorizer_model
            partial_fit: Whether to run `partial_fit` for online learning

        Returns:
            tf_idf: The resulting matrix giving a value (importance score) for each word per topic
            words: The names of the words to which values were given
        """
        documents = self._preprocess_text(documents_per_topic.text.values)

        if partial_fit:
            X = self.vectorizer_model.partial_fit(documents).update_bow(
                documents
            )
        elif fit:
            self.vectorizer_model.fit(documents)
            X = self.vectorizer_model.transform(documents)
        else:
            X = self.vectorizer_model.transform(documents)

        # Scikit-Learn Deprecation: get_feature_names is deprecated in 1.0
        # and will be removed in 1.2. Please use get_feature_names_out instead.
        # if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = self.vectorizer_model.get_feature_names_out()
        # else:
        #    words = self.vectorizer_model.get_feature_names()

        if fit:
            self.ctfidf_model = self.ctfidf_model.fit(X)

        c_tf_idf = self.ctfidf_model.transform(X)

        return c_tf_idf, words

    def _extract_words_per_topic(
        self,
        words: List[str],
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix = None,
        calculate_aspects: bool = True,
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Based on tf_idf scores per topic, extract the top n words per topic

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Taken from: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            documents: DataFrame with documents and their topic IDs
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            c_tf_idf = self.c_tf_idf_

        labels = sorted(list(documents.component_id.unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        top_n_words = max(self.top_n_words, 30)
        indices = self._top_n_idx_sparse(c_tf_idf, top_n_words)
        scores = self._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        topics = {
            label: [
                (
                    (words[word_index], score)
                    if word_index is not None and score > 0
                    else ("", 0.00001)
                )
                for word_index, score in zip(
                    indices[index][::-1], scores[index][::-1]
                )
            ]
            for index, label in enumerate(labels)
        }

        # Fine-tune the topic representations
        topics = {
            label: values[: self.top_n_words]
            for label, values in topics.items()
        }

        return topics

    def get_topic_info(self, topic=None):
        """Get information about each topic including its ID, frequency, and name.

        Taken from: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/_bertopic.py

        Arguments:
            topic: A specific topic for which you want the frequency

        Returns:
            info: The information relating to either a single topic or all topics

        Examples:

        ```python
        info_df = topic_model.get_topic_info()
        ```
        """
        info = pd.DataFrame(
            self.topic_sizes_.items(), columns=["component_id", "count"]
        )
        info["name"] = info.component_id.map(self.topic_labels_)

        # Main Keywords
        values = {
            topic: list(list(zip(*values))[0])
            for topic, values in self.topic_representations_.items()
        }
        info["representation"] = info["component_id"].map(values)

        # Select specific topic to return
        if topic is not None:
            info = info.loc[info.component_id == topic, :]

        return info.reset_index(drop=True)

    def get_topic(self, topic):
        """Return top n words for a specific topic and their c-TF-IDF scores

        Arguments:
            topic: A specific topic for which you want its representation

        Returns:
            The top n words for a specific word and its respective c-TF-IDF scores

        Examples:

        ```python
        topic = topic_model.get_topic(12)
        ```
        """
        if topic in self.topic_representations_:
            return self.topic_representations_[topic]
        else:
            return False

    def _update_topic_size(self, documents: pd.DataFrame):
        """Calculate the topic sizes

        Arguments:
            documents: Updated dataframe with documents and their corresponding IDs and newly added Topics
        """
        self.topic_sizes_ = collections.Counter(
            documents.component_id.values.tolist()
        )
        self.topics_ = documents.component_id.astype(int).tolist()

    @staticmethod
    def _top_n_idx_sparse(matrix: csr_matrix, n: int) -> np.ndarray:
        """Return indices of top n values in each row of a sparse matrix

        Retrieved from:
            https://stackoverflow.com/questions/49207275/finding-the-top-n-values-in-a-row-of-a-scipy-sparse-matrix

        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            n: The number of highest values to extract from each row

        Returns:
            indices: The top n indices per row
        """
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            values = matrix.indices[
                le
                + np.argpartition(matrix.data[le:ri], -n_row_pick)[
                    -n_row_pick:
                ]
            ]
            values = [
                values[index] if len(values) >= index + 1 else None
                for index in range(n)
            ]
            indices.append(values)
        return np.array(indices)

    @staticmethod
    def _top_n_values_sparse(
        matrix: csr_matrix, indices: np.ndarray
    ) -> np.ndarray:
        """Return the top n values for each row in a sparse matrix

        Arguments:
            matrix: The sparse matrix from which to get the top n indices per row
            indices: The top n indices per row

        Returns:
            top_values: The top n scores per row
        """
        top_values = []
        for row, values in enumerate(indices):
            scores = np.array(
                [
                    matrix[row, value] if value is not None else 0
                    for value in values
                ]
            )
            top_values.append(scores)
        return np.array(top_values)


class ClassTfidfTransformer(TfidfTransformer):
    """
    A Class-based TF-IDF procedure using scikit-learns TfidfTransformer as a base.

    Code taken from original BERTopic implementation by Martin Grootendorst.

    See: https://github.com/MaartenGr/BERTopic/blob/master/bertopic/vectorizers/_ctfidf.py


    ![](../algorithm/c-TF-IDF.svg)

    c-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class. Thus, each class is converted to a single document
    instead of set of documents. The frequency of each word **x** is extracted
    for each class **c** and is **l1** normalized. This constitutes the term frequency.

    Then, the term frequency is multiplied with IDF which is the logarithm of 1 plus
    the average number of words per class **A** divided by the frequency of word **x**
    across all classes.

    Arguments:
        bm25_weighting: Uses BM25-inspired idf-weighting procedure instead of the procedure
                        as defined in the c-TF-IDF formula. It uses the following weighting scheme:
                        `log(1+((avg_nr_samples - df + 0.5) / (df+0.5)))`
        reduce_frequent_words: Takes the square root of the bag-of-words after normalizing the matrix.
                               Helps to reduce the impact of words that appear too frequently.
        seed_words: Specific words that will have their idf value increased by
                    the value of `seed_multiplier`.
                    NOTE: This will only increase the value of words that have an exact match.
        seed_multiplier: The value with which the idf values of the words in `seed_words`
                         are multiplied.

    Examples:

    ```python
    transformer = ClassTfidfTransformer()
    ```
    """

    def __init__(
        self,
        bm25_weighting: bool = False,
        reduce_frequent_words: bool = True,
        seed_words: List[str] = None,
        seed_multiplier: float = 2,
    ):
        self.bm25_weighting = bm25_weighting
        self.reduce_frequent_words = reduce_frequent_words
        self.seed_words = seed_words
        self.seed_multiplier = seed_multiplier
        super(ClassTfidfTransformer, self).__init__()

    def fit(self, X: sp.csr_matrix, multiplier: np.ndarray = None):
        """Learn the idf vector (global term weights).

        Arguments:
            X: A matrix of term/token counts.
            multiplier: A multiplier for increasing/decreasing certain IDF scores
        """
        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64

        if self.use_idf:
            _, n_features = X.shape

            # Calculate the frequency of words across all classes
            df = np.squeeze(np.asarray(X.sum(axis=0)))

            # Calculate the average number of samples as regularization
            avg_nr_samples = int(X.sum(axis=1).mean())

            # BM25-inspired weighting procedure
            if self.bm25_weighting:
                idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))

            # Divide the average number of samples by the word frequency
            # +1 is added to force values to be positive
            else:
                idf = np.log((avg_nr_samples / df) + 1)

            # Multiplier to increase/decrease certain idf scores
            if multiplier is not None:
                idf = idf * multiplier

            self._idf_diag = sp.diags(
                idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X: sp.csr_matrix):
        """Transform a count-based matrix to c-TF-IDF

        Arguments:
            X (sparse matrix): A matrix of term/token counts.

        Returns:
            X (sparse matrix): A c-TF-IDF matrix
        """
        if self.use_idf:
            X = normalize(X, axis=1, norm="l1", copy=False)

            if self.reduce_frequent_words:
                X.data = np.sqrt(X.data)

            X = X * self._idf_diag

        return X


class ChineseTokenizer:

    def __init__(self, stopwords=[], max_len=1000):
        """
        Initialize the Chinese Tokenizer.
        """
        self.stopwords = stopwords
        self.max_len = max_len

    def tokenize(self, text):
        """
        Tokenize a text for representation.
        """
        if len(text) > self.max_len:
            texts = []
            i = 0
            j = self.max_len
            while j < len(text):
                if text[j] == " ":
                    texts.append(text[i:j])
                    i = j
                    j += self.max_len
                else:
                    j += 1
            texts.append(text[i:-1])
        else:
            texts = [text]

        tokenized = []
        for t in texts:
            doc = jieba.lcut(t)
            for token in doc:
                # only allow word-tokens as representation
                if re.match(r"^[\u4e00-\u9fffA-Za-z]+$", token):
                    if token not in self.stopwords:
                        tokenized.append(token)
        return tokenized

    def __call__(self, text):
        return self.tokenize(text)


class EnglishTokenizer:

    def __init__(self, stopwords=[]):
        """
        Initialize the English Tokenizer.
        """
        self.stopwords = stopwords

    def tokenize(self, text):
        """
        Tokenize a text for representation.
        """
        regex = r"((?<=\s)\b\w[\w-]+\b|\b\w[\w-]+\b)"
        tokens = re.findall(regex, text)
        tokens = [t.lower() for t in tokens]
        # remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def __call__(self, text):
        return self.tokenize(text)


class GenericTokenizer:

    def __init__(self, stopwords=[]):
        """
        Initialize the Generic Tokenizer.
        """
        self.stopwords = stopwords

    def tokenize(self, text):
        """
        Tokenize a text for representation.
        """
        tokens = text.split(" ")
        # remove non-alphanumeric characters
        tokens = [
            re.sub(r"[,.;@#?!&$_\/\ƙ\<>\-+#*=(){}]", "", t) for t in tokens
        ]
        # remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def __call__(self, text):
        return self.tokenize(text)


