# FOS (Future Of Science) Benchmark

A reproducible pipeline to build graph edges and node features for the FOS benchmark.

## If you want to use our pre-built FOS_Art&Business benchmark:

You can download the benchmark at [our HuggingFace dataset](https://huggingface.co/datasets/Morteza24/future-of-science) and place the `Art_Business/` directory here. You won't need to perform steps 1 and 2.

## If you want to build your own FOS benchmark:

If you haven't already prepared the required artifacts from the OpenAlex knowledge graph, head over to [this page](https://github.com/kiyan-rezaee/future-of-science/blob/main/OpenAlex_Knowledge_Graph) and then proceed with the following steps:

### Step 1 — Preparing edges

1. Open [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml) and configure database connection, historical time span (start & end years), and included domains.

2. Run `edge_creation.py`.

### Step 2 — Preparing node features

1. Run `node_metadata_collection.py`.

2. Run `node_feature_creation.py`.

## Notes

- Keep a consistent `config.yaml` across scripts to avoid mismatches.
- Use a small list of domains for tractability.
