import pandas as pd
import psycopg
import yaml
import os


config_path = os.path.join("..", "config.yaml")
with open(config_path, "rt") as config_file:
	config = yaml.safe_load(config_file)

out_dir = os.path.join("./edges", "_".join(config["DOMAINS"]))
os.makedirs(out_dir, exist_ok=True)

try:
	with psycopg.connect(host=config["DB_HOST"], port=config["DB_PORT"], dbname=config["DB_NAME"], user=config["DB_USER"], password=config["DB_PASSWORD"]) as conn:
		with conn.cursor() as cur:
			for year in range(config["START_YEAR"], config["END_YEAR"] + 1):
				table = f'{"_".join(config["DOMAINS"])}_{year}_concept_pairs'
				file_path = os.path.join(out_dir, f"{year}.csv")
				query = f"COPY {table} TO '{file_path}' WITH (FORMAT csv);"
				cur.execute(query)
				print(f"Exported {table} -> {file_path}")
finally:
	conn.close()

print("Combining all yearly edge files into a single file...")

edges_list = []
for year in range(config["START_YEAR"], config["END_YEAR"]+1):
    edges = pd.read_csv(os.path.join(out_dir, f"{year}.csv"), header=None)
    edges_list.append(edges)
edges = pd.concat(edges_list)
edges = edges.reset_index().drop("index", axis=1)
edges = edges.sort_values(by='ts').reset_index(drop=True)
edges.to_csv(os.path.join(out_dir, "all_edges.csv"), index=False, header=False)

print("All edges combined and saved to all_edges.csv")
