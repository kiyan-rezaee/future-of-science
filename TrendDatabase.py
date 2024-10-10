import os
import glob
import psycopg2
import json
from itertools import combinations



def load_concept_pairs(folder_path):
    """
    Loads concept pairs from JSON files into a PostgreSQL database.

    Args:
        folder_path (str): The path to the folder containing the JSON files.
    """

    json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)

    with psycopg2.connect("host=localhost dbname=TrendDB user=Trend password=2424") as conn:
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

                # Insert concept pairs into the database
                for concept_group in data:
                    for concept1, concept2 in combinations(sorted(concept_group), 2):
                        cur.execute(f"INSERT INTO concept_pairs VALUES ('{concept1}', '{concept2}', {year}) ON CONFLICT DO NOTHING")
                        conn.commit()

                # Print progress
                print(f"\rFile {i+1}/{len(json_files)} finished: {json_file}", end="\n\n")