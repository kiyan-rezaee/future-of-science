# OpenAlex Knowledge Graph

We construct the FOS dataset using two primary artifacts from the OpenAlex knowledge graph:

#### 1. The Concepts Taxonomy:
This is the `concepts_taxonomy.csv` file at the current directory. For reference, you can find the link to the original file on the OpenAlex website using [this link](https://docs.openalex.org/api-entities/concepts).

#### 2. The Publication Corpus:
Building the publication corpus consists of three steps:

1. Collecting the Publication Corpus
	- Run `publication_corpus_collection.py` to gather publication metadata from the OpenAlex API.
	- By default the script collects the full historical span (1827-2025) which is time- and resource-intensive. You can [edit `config.yaml` and set your desired historical span](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml#L8).

2. Creating Node Lists
	- Run `node_creation.py` to generate node lists for the 19 root domains.
	- These node files are used to filter the publication corpus by domain.

3. Building the Per-Year Publication Corpus
	- You should run the `publication_corpus_creation.py` script which builds a database table for each year in your desired historical span in parallel. You can set the number of workers in [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml#L45).
	- This script requires a running PostgerSQL server. Open the [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml) file and set your database connection info, along with historical span (start and end year) and domains to include in the dataset.
	- By default, the config file includes 2 domains, Art and Business. including too many domain would require too much time and resources. It is recommended to include only a small set of domains.

### Next Step

After preparing the above artifacts, move on to [this page](https://github.com/kiyan-rezaee/future-of-science/tree/main/FOS_Benchmark) to create your FOS Benchmark.

### TODO:

- Only collect selected domains in publication_corpus_collection
- Include a pre-collected publication corpus
- Include pre-processed node data
- Include pre-built datasets for a subset of domains
