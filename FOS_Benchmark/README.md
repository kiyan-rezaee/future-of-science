# FOS (Future Of Science) Benchmark

A reproducible pipeline to build graph edges and node features for the FOS benchmark.

## Prerequisite
- If you haven't already prepared the required artifacts from the OpenAlex knowledge graph, head over to [this page](https://github.com/kiyan-rezaee/future-of-science/blob/main/OpenAlex_Knowledge_Graph).

## Step 1 — Preparing edges

1. Open [`config.yaml`](https://github.com/kiyan-rezaee/future-of-science/blob/main/config.yaml) and configure database connection, historical time span (start & end years), and included domains.

2. Run `edge_creation.py`.

## Step 2 — Preparing node features

1. Run `node_metadata_collection.py`.

2. Run `node_feature_creation.py`.

## Notes

- Keep a consistent `config.yaml` across scripts to avoid mismatches.
- Use a small list of domains for tractability.
