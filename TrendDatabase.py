import psycopg2
import json
from itertools import combinations
import os
import glob

def find_json_files(folder_path):
    json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
    return json_files

folder_path = './'

json_files = find_json_files(folder_path)

conn = psycopg2.connect(
    host="your_host",
    database="your_database",
    user="your_username",
    password="your_password"
)
cur = conn.cursor()

cur.execute('''
    CREATE TABLE IF NOT EXISTS concept_pairs (
        Concept1 VARCHAR(255),
        Concept2 VARCHAR(255),
        year INTEGER,
        PRIMARY KEY (Concept1, Concept2, year)
    )
''')
conn.commit()

def load_existing_pairs(year):
    cur.execute('''
        SELECT Concept1, Concept2 FROM concept_pairs WHERE year = %s
    ''', (year,))
    existing_pairs = cur.fetchall()
    return set((min(c1, c2), max(c1, c2)) for c1, c2 in existing_pairs)


def insert_concept_pairs(concept_list, year, existing_pairs):
    for concept1, concept2 in combinations(sorted(concept_list), 2):
        pair = (concept1, concept2)
        if pair not in existing_pairs:
            cur.execute('''
                INSERT INTO concept_pairs (Concept1, Concept2, year)
                VALUES (%s, %s, %s)
            ''', (concept1, concept2, year))
            existing_pairs.add(pair)
            conn.commit()



for json_file in json_files:
    year = int(json_file[2:6])
    with open(json_file, 'r') as f:
        data = json.load(f)
    for concept_group in data:
        insert_concept_pairs(concept_group, year)
    print(f"{json_file} is done!")    

cur.close()
conn.close()