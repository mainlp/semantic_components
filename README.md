

[![PyPI - Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://pypi.org/project/semantic-components/0.1.0/)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/mainlp/semantic_components/blob/main/LICENSE)
[![PyPI - PyPi](https://img.shields.io/pypi/v/semantic-components)](https://pypi.org/project/semantic-components/0.1.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2410.21054-<COLOR>.svg)](https://arxiv.org/abs/2410.21054)


# Semantic Component Analysis: Discovering Patterns in Short Texts Beyond Topics

<img src="images/tweet_decomposition.png" width="50%" height="50%" align="right" />

Semantic Component Analysis (SCA) is a powerful tool to analyse *your* text datasets. If you want to find out how it works and why it is the right tool for you, consider reading [our paper](https://arxiv.org/abs/2410.21054).

If you just want to test the method as quickly as possible, continue with the Quick Start section. For everything else, the Manual Installation section should have you covered. If you run into any problems or have suggestions, feel free to create an issue and we will try to adress it in future releases.

## Quick Start

The method is available through `pypi` and part of the `semantic_components` package. You can install it as

```bash
pip install semantic_components
```

Running SCA is as simple as importing this package, and two lines for instantiating and fitting the model:

```python
from semantic_components.sca import SCA

# fit sca model to data
sca = SCA(alpha_decomposition=0.1, mu=0.9, combine_overlap_threshold=0.5)
scores, residuals, ids = sca.fit(documents, embeddings)

# get representations and explainable transformations
representations = sca.representations  # pandas df
transformed = sca.transform(embeddings)  # equivalent to variable scores above
```

A full example including computing the embeddings and loading the Trump dataset can be found in `example.py`. We advise to clone this repository, if you want to run this example and/or our experiments found in the `experiments/` directory. 

Where applicable, the `results/` folder is where the experiment scripts will store their results. Run SCA with `save_results=True` and `verbose=True` to enable this behaviour. This will generate a `reports.txt` containing information and evaluation metrics. Furthermore, there will be `.pkl` and `.txt` files with the representations of the semantic components found by the procedure.

## OCTIS Evaluation

By default, we do not install `octis` as it requires older versions of some other packages and thus creates compatibility issues. If you want to 
use OCTIS evaluation (i.e. topic coherence and diversity), consider installing this package as
```bash
pip install semantic_components[octis]
```
for both versions, we recommend Python 3.10 or higher.

## Manual Installation

In order to run the code provided in this repository, a number of non-standard Python packages need to be installed. As of October 2024, Python 3.10.x with the most current versions should work with the implementions provided. Here is a pip install command you can use in your environment to install all of them.
```bash
pip install sentence_transformers umap-learn hdbscan jieba scikit-learn pandas octis
```

Our experiments have been run with the following versions:

```
hdbscan                  0.8.39
jieba                    0.42.1
numpy                    1.26.4
octis                    1.14.0
pandas                   2.2.3
scikit-learn             1.1.0
sentence-transformers    3.2.0
torch                    2.4.1
transformers             4.45.2
umap-learn               0.5.6
```
You can clone this repository to your machine as follows:
```bash
git clone git@github.com:eichinflo/semantic_components.git
```

If you work with conda for example, you can run the following commands to get an environment suited to run the code:

```bash
cd semantic_components
conda create -n sca python=3.10.15
conda activate sca
pip install sentence_transformers umap-learn hdbscan jieba scikit-learn pandas octis
```
then, you're ready to run the example script which reproduces part of the results on the Trump dataset:
```bash
python example.py
```

## Data

All data used in this work is publicly available. The Trump dataset is available from [the Trump Twitter Archive](https://www.thetrumparchive.com/). You can choose to download your version as `.csv` directly from that page and put it in the `data/`
directory (to work with the experiment code, rename it to `trump_tweets.csv`). However, we provide the version we used in this repository. 

Besides, we publish the Chinese News dataset, which we acquired to the Twitter API and was kept updated until our academic access got revoked in April 2023. We provide it as a download [HERE](https://drive.google.com/drive/folders/19H4gjnXGviXZUS8prv3l1WngGKwOpCMP?usp=sharing).

The current version of the Hausa Tweet dataset is available at the [NaijaSenti repository](https://github.com/hausanlp/NaijaSenti/blob/main/sections/unlabeled_twitter_corpus.md).

## Support of Other Languages

We wanted to make `SCA` as adaptable as possible to use-cases in other languages. The main parts of the pipeline that do not always generalize across languages are the base embedding model as well as the tokenizer and stopwords-list used to compute the c-TF-IDF representations. Since the embeddings are calculated outside of the `SCA` class, you can use any method that will output vector-valued embeddings. Just make sure to pass them as `numpy.array` or equivalent.

For the other parts, you can pass custom versions when instantiating `SCA`:

```python
from semantic_components.sca import SCA
from semantic_components.representation import GenericTokenizer

custom_tokenizer = GenericTokenizer()
custom_stopwords_path = "path/to/stopwords.txt"

# fit sca model to data
sca = SCA(tokenizer=custom_tokenizer, stopwords_path=custom_stopwords_path)
scores, residuals, ids = sca.fit(documents, embeddings)
```

You can look at the implementation of `GenericTokenizer` for a minimal example of what your custom tokenizer should do (you only need to implement `tokenize` and `__call__`). The stopwords are passed as a path to a stopwords file where each line is interpreted as a single stopword. The representer will ignore these words when calculating the token representations. Passing either of these arguments will overwrite the respective standard choices inferred by the `language` argument (which currently only supports Chinese and English, though the latter generalizes to other languages where tokens are separated by whitespace).

## AI Usage Disclaimer

The code in this repository has been written with the support of code completions of an AI coding assistant, namely GitHub Copilot. Completions were mostly single lines up to a few lines of code and were always checked carefully to ensure their functionality and safety. Furthermore, we did our best to avoid accepting code completions that would be incompatible with the license of our code or could be regarded as plagiarism.


## Acknowledgements

We're grateful to Kristin Shi-Kupfer and David Adelani for consulting on the Chinese and Hausa datasets respectively. Furthermore, we would like to mention that the code of the `c-TF-IDF` representer has been largely adapted from the original [BERTopic](https://github.com/MaartenGr/BERTopic) implementation by 
Maarten Grootendorst released under MIT license. 

## Citing This Work

If you're using this work for your project, please consider citing our paper:

```bibtext
@misc{eichin2024semanticcomponentanalysisdiscovering,
      title={Semantic Component Analysis: Discovering Patterns in Short Texts Beyond Topics}, 
      author={Florian Eichin and Carolin Schuster and Georg Groh and Michael A. Hedderich},
      year={2024},
      eprint={2410.21054},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.21054}, 
}
```
