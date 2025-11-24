# OpenAlex Knowledge Graph

We construct the FOS dataset using two primary artifacts from the OpenAlex knowledge graph:

#### 1. The Concepts Taxonomy:
This is the `concepts_taxonomy.csv` file at the current directory. For reference, you can find the link to the original file on the OpenAlex website using [this link](https://docs.openalex.org/api-entities/concepts).

#### 2. The Publication Corpus:
Building the publication corpus consists of three steps:

1. Collecting the Publication Corpus
	- Run `publication_corpus_collection.py` to gather publication metadata from the OpenAlex API.
	- By default the script collects the full historical span (1827-2025) which is time- and resource-intensive. You can [edit the script and set your desired historical span](https://github.com/kiyan-rezaee/future-of-science/blob/main/OpenAlex_Knowledge_Graph/publication_corpus_collection.py#L116).

2. Creating Node Lists
	- Run `node_creation.py` to generate node lists for the 19 root domains.
	- These node files are used to filter the publication corpus by domain.

3. Building the Per-Year Publication Corpus
	- You should run the `publication_corpus_creation.py` script for each year in your desired historical span.
	- This script requires a running PostgerSQL server. Open the file and update the [Configurations](https://github.com/kiyan-rezaee/future-of-science/blob/main/publication_corpus_creation#L10) at the top with your DB connection info, the target `year`, and `domains` to include in the dataset.
	- By default, the script includes all the 19 domains, but creating such dataset would take too much time and resource. It is recommended to include only a small set of domains e.g. `["Computer_Science", "Biology"]`.

### Next Step

After preparing the above artifacts, move on to [this page](https://github.com/kiyan-rezaee/future-of-science/tree/main/FOS_Benchmark) to create your FOS Benchmark.

### TODO:

- Include a pre-collected publication corpus
- Include pre-processed node data
- Include pre-built datasets for a subset of domains
