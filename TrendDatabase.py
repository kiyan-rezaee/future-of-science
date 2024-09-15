from itertools import combinations
import os, glob
import psycopg
import json


folder_path = "json_files"
json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
len_files = len(json_files)


with psycopg.connect("host=localhost dbname=TrendDB user=Trend password=2424") as conn:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS concept_pairs(
                Concept1 VARCHAR(255) NOT NULL,
                Concept2 VARCHAR(255) NOT NULL,
                year INTEGER,
                PRIMARY KEY (Concept1, Concept2, year))
            """)
        for i, json_file in enumerate(json_files):
            year = int(json_file.split("/")[-1][:4])
            with open(json_file, 'rt') as f:
                data = json.load(f)
            len_data = len(data)
            for j, concept_group in enumerate(data):
                for concept1, concept2 in combinations(sorted(concept_group), 2):
                    cur.execute(f"INSERT INTO concept_pairs VALUES ('{concept1}', '{concept2}', {year}) ON CONFLICT DO NOTHING")
                    conn.commit()
                if (j+1) % 10 == 0:
                    print(f"{j+1}/{len_data}")
            print(f"file {i+1}/{len_files} finished:", json_file)
            print()

