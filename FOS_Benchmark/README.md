# FOS (Future Of Science) Benchmark

A reproducible pipeline to build graph edges and node features for the FOS benchmark.

## Prerequisite
- If you haven't already prepared the required artifacts from the OpenAlex knowledge graph, head over to [this page](https://github.com/kiyan-rezaee/future-of-science/blob/main/OpenAlex_Knowledge_Graph).

## Step 1 — Preparing edges

1. Open the `edge_creation.py` script and configure database connection, historical time span (start & end years), and included domains.

2. Run `edge_creation.py`.

## Step 2 — Preparing node features

1. Open the `node_metadata_collection.py` script and edit the global `DOMAINS` variable to match the domains you want to include.

2. Run `node_metadata_collection.py`.

3. Open the `node_feature_creation.py` script and edit the global `DOMAINS` variable to match the domains you want to include.

2. Run `node_feature_creation.py`.

## Notes

- Keep a consistent `DOMAINS` list across scripts to avoid mismatches.
- Use a small list of domains for tractability.
