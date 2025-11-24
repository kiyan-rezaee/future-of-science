import os
import glob
import json
import shutil  # see line 96
import psycopg
import pandas as pd
from itertools import combinations


# Configurations
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "PublicationCorpus"
DB_USER = "Enter Your User"
DB_PASSWORD = "Enter Your Password"
YEAR = "2022"
DOMAINS = ["Political_Science", "Philosophy", "Economics", "Business", "Psychology", "Mathematics", "Medicine",
          "Biology", "Computer_Science", "Geology", "Chemistry", "Art", "Sociology", "Engineering", "Geography",
          "History", "Materials_Science", "Physics", "Environmental_Science"]


###############################################
# Process JSONs : Process JSON files and insert data into PostgreSQL
# Input         : None (fetches JSON files from a directory)
# Outputs       : None (inserts into database and prints progress)
###############################################
def process_json_files(year, fields):
    nodes = set()
    for field in fields:
        nodes |= set(pd.read_csv(f"./nodes/{field}.csv").values.flatten())

    #------------------------------------------
    # Block to Load JSON Files
    # Fetches all .json files from the specified folder path
    #------------------------------------------
    # Define the folder path and retrieve JSON files
    folder_path = f"./json_files/{year}/"
    json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)

    # Get the total number of files
    len_files = len(json_files)

    #------------------------------------------
    # Block to Connect to PostgreSQL Database
    # Establishes a connection to the PostgreSQL database and ensures the table exists
    #------------------------------------------
    # Connect to PostgreSQL and create table if it doesn't exist
    with psycopg.connect(f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}") as conn:
        with conn.cursor() as cur:
            # Create the concept_pairs table if it does not exist
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {"_".join(fields)}_{year}_concept_pairs(
                    Concept1 VARCHAR(255) NOT NULL,
                    Concept2 VARCHAR(255) NOT NULL,
                    year INTEGER,
                    PRIMARY KEY (Concept1, Concept2, year))
            """)

            #------------------------------------------
            # Block to Process Each JSON File
            # Iterates through each file, reads data, and inserts into the database
            #------------------------------------------
            # Process each JSON file
            for i, json_file in enumerate(json_files):
                # Extract year from file path
                year = int(json_file.split("/")[-1][:4])

                # Open and load the JSON data from file
                with open(json_file, 'rt') as f:
                    data = json.load(f)
                data = [intersection for intersection in (set(group) & nodes for group in data) if len(intersection) > 1]
                len_data = len(data)

                #------------------------------------------
                # Block to Insert Data into Database
                # Loops through concept groups and inserts pair combinations
                #------------------------------------------
                # Iterate through each group of concepts in the data
                for j, concept_group in enumerate(data):
                    # Get combinations of two concepts, ordered alphabetically
                    for concept1, concept2 in combinations(sorted(concept_group), 2):
                        # Insert concept pairs into the database
                        cur.execute(f"INSERT INTO {"_".join(fields)}_{year}_concept_pairs VALUES ('{concept1}', '{concept2}', {year}) ON CONFLICT DO NOTHING")

                        # Commit the transaction
                        conn.commit()

                    # Print progress every 10 entries
                    if (j+1) % 10 == 0:
                        print(f"\r{j+1}/{len_data}", end="")

                # Print file processing completion message
                print(f"\nfile {i+1}/{len_files} finished:", json_file, end="\n\n")
                print()

                # You can uncomment the following line to move processed files to a 'done' folder
                # This makes stopping and resuming the script easier
                # shutil.move(json_file, "./processed_json_files/")


if __name__ == "__main__":
    process_json_files(YEAR, DOMAINS)
