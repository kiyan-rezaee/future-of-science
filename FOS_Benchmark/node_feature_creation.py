import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DOMAINS = ["Political_Science", "Philosophy", "Economics", "Business", "Psychology", "Mathematics", "Medicine",
          "Biology", "Computer_Science", "Geology", "Chemistry", "Art", "Sociology", "Engineering", "Geography",
          "History", "Materials_Science", "Physics", "Environmental_Science"]


nodes_dir = "../OpenAlex_Knowledge_Graph/nodes/"
nodes = set()
for field in DOMAINS:
    df = pd.read_csv(nodes_dir + field + ".csv", header=None)
    nodes |= set(df.iloc[:, 0])


# Load a model trained on scientific texts
model = SentenceTransformer('sentence-transformers/allenai-specter')


def get_embedding(text):
    if text is None or text.strip() == "":
        return np.zeros(model.get_sentence_embedding_dimension())
    return model.encode(text)


def positional_encoding(level, dim):
    pe = np.zeros(dim)
    for i in range(0, dim, 2):
        div_term = np.exp(i * -np.log(10000.0) / dim)
        pe[i] = np.sin(level * div_term)
        if i + 1 < dim:
            pe[i + 1] = np.cos(level * div_term)
    return pe


def encode_concept(concept):
    # Embed the main description
    desc_embedding = get_embedding(concept['description'])

    # Embed the display name
    name_embedding = get_embedding(concept['display_name'])

    # Embed ancestors and average
    ancestor_embeddings = [get_embedding(ancestor) for ancestor in concept.get('ancestors', [])]
    if ancestor_embeddings:
        ancestor_avg = np.mean(ancestor_embeddings, axis=0)
    else:
        ancestor_avg = np.zeros(desc_embedding.shape)

    # Embed related concepts and average
    related_embeddings = [get_embedding(rel) for rel in concept.get('related_concepts', [])]
    if related_embeddings:
        related_avg = np.mean(related_embeddings, axis=0)
    else:
        related_avg = np.zeros(desc_embedding.shape)

    # Encode level using sinusoidal positional encoding
    level_encoding = positional_encoding(concept['level'], desc_embedding.shape[0])

    # Combine all embeddings
    none = desc_embedding + name_embedding + ancestor_avg + related_avg + level_encoding
    combined_desc = name_embedding + ancestor_avg + related_avg + level_encoding
    combined_name = desc_embedding + ancestor_avg + related_avg + level_encoding
    combined_ancestor = desc_embedding + name_embedding + related_avg + level_encoding
    combined_related = desc_embedding + name_embedding + ancestor_avg + level_encoding
    combined_level = desc_embedding + name_embedding + ancestor_avg + related_avg

    # return combined
    return {"none": none, "desc": combined_desc, "name": combined_name, "ancestor": combined_ancestor, "related": combined_related, "level": combined_level}


features = ["none", "desc", "name", "ancestor", "related", "level"]
df = {feature: pd.DataFrame(columns=['node_id', 'embeddings']) for feature in features}
out_dir = os.path.join('node_embeddings', "_".join(DOMAINS))
os.makedirs(out_dir, exist_ok=True)
data_dir = os.path.join('node_data', "_".join(DOMAINS))
if not os.path.isdir(data_dir):
    print(f"node data directory not found: {data_dir}")
else:
    for filename in sorted(os.listdir(data_dir)):
        if not filename.startswith("node_data_part_") or not filename.endswith(".json"):
            continue
        part = filename[len("node_data_part_"):-len(".json")]
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        data = [item for item in data if item["id"].split("/")[-1] in nodes]
        print(f"Part {part}")
        for concept in data:
            combined = encode_concept(concept)
            for feature in features:
                new_row = pd.Series({'node_id': concept['id'].split("/")[-1], 'embeddings': combined[feature]})
                df[feature] = pd.concat([df[feature], new_row.to_frame().T], ignore_index=True)
    for feature in features:
        df[feature].reset_index(drop=True, inplace=True)
        if feature == "none":
            df[feature].to_pickle(os.path.join(out_dir, f'full_features.pkl'))
        else:
            df[feature].to_pickle(os.path.join(out_dir, f'features_without_{feature}.pkl'))

print("Node feature creation completed.")
