import pandas as pd
import psycopg
import os


# Configurations
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "PublicationCorpus"
DB_USER = "Enter Your User"
DB_PASSWORD = "Enter Your Password"
START_YEAR = 1827
END_YEAR = 2025
DOMAINS = ["Political_Science", "Philosophy", "Economics", "Business", "Psychology", "Mathematics", "Medicine",
          "Biology", "Computer_Science", "Geology", "Chemistry", "Art", "Sociology", "Engineering", "Geography",
          "History", "Materials_Science", "Physics", "Environmental_Science"]


out_dir = os.path.join("./edges", "_".join(DOMAINS))
os.makedirs(out_dir, exist_ok=True)

try:
	with psycopg.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD) as conn:
		with conn.cursor() as cur:
			for year in range(START_YEAR, END_YEAR + 1):
				table = f'{"_".join(DOMAINS)}_{year}_concept_pairs'
				file_path = os.path.join(out_dir, f"{year}.csv")
				query = f"COPY {table} TO {file_path} WITH (FORMAT csv);"
				cur.execute(query)
				print(f"Exported {table} -> {file_path}")
finally:
	conn.close()

print("Combining all yearly edge files into a single file...")

edges_list = []
for year in range(START_YEAR, END_YEAR+1):
    edges = pd.read_csv(os.path.join(out_dir, f"{year}.csv"))
    edges_list.append(edges)
edges = pd.concat(edges_list)
edges = edges.reset_index().drop("index", axis=1)
edges = edges.rename(columns={0: 'src', 1: 'dst', 2: 'ts'})
edges = edges.sort_values(by='ts').reset_index(drop=True)
edges.to_csv(os.path.join(out_dir, "all_edges.csv"), index=False)

print("All edges combined and saved to all_edges.csv")
