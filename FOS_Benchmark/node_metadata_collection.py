from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import requests
import json
import math
import time
import os


DOMAINS = ["Political_Science", "Philosophy", "Economics", "Business", "Psychology", "Mathematics", "Medicine",
          "Biology", "Computer_Science", "Geology", "Chemistry", "Art", "Sociology", "Engineering", "Geography",
          "History", "Materials_Science", "Physics", "Environmental_Science"]


def create_session():
    """
    Creates a requests session with retry strategy and custom headers.
    """
    session = requests.Session()

    # Define retry strategy
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=1,  # Exponential backoff factor (e.g., 1, 2, 4, 8, 16 seconds)
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Methods to retry
    )

    # Mount the HTTPAdapter with the retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set headers to mimic a real browser
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/66.0.3359.139 Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive"
    })

    return session


def call_api(url):
    """
    Makes a GET request to the specified URL with retry logic.
    """
    session = create_session()
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        try:
            return response.json()
        except ValueError:
            return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None


def main_results(url):
    data = call_api(url)
    if data is not None:
        if isinstance(data, dict):
            ancestors = [ancestor['display_name'] for ancestor in data.get('ancestors', [])]
            related_concepts = [concept['display_name'] for concept in data.get('related_concepts', [])]
            return (
                data.get('id'),
                data.get('display_name'),
                data.get('level'),
                data.get('description'),
                ancestors,
                related_concepts
            )
        else:
            print(data)


def paper_counts(url):
    data = call_api(url)
    if data is not None:
        if isinstance(data, dict):
            return {year["key"]: year["count"] for year in data.get("group_by", [])}
        else:
            print(data)


def convert_openalex_url(url):
    """
    Converts an OpenAlex concept web URL to the corresponding API URLs.
    """
    # Extract the concept ID from the URL (everything after the last '/')
    id = url.split('/')[-1]

    # Create the concept API URL by capitalizing the first letter of the ID
    concept_api = f"https://api.openalex.org/concepts/C{id[1:]}"

    # Create the works API URL using the original ID and fixed query parameters
    works_api = f"https://api.openalex.org/works?group_by=publication_year&per_page=200&filter=concepts.id:{id}"

    return concept_api, works_api


def dictionary_entry(main_url):
    row = {}
    url1, url2 = convert_openalex_url(main_url)
    main_data = main_results(url1)
    if main_data:
        (row['id'], row['display_name'], row['level'],
         row['description'], row['ancestors'], row['related_concepts']) = main_data
    row['number_of_papers'] = paper_counts(url2)
    return row


nodes = set()
for field in DOMAINS:
	nodes |= set(pd.read_csv(f"../OpenAlex_Knowledge_Graph/nodes/{field}.csv").values.flatten())

openalex_ids = ["https://openalex.org/" + node.lower() for node in nodes]

chunk_size = math.ceil(len(openalex_ids) / 10)

chunks = [openalex_ids[i:i + chunk_size] for i in range(0, len(openalex_ids), chunk_size)]

out_dir = os.path.join('node_data', "_".join(DOMAINS))
os.makedirs(out_dir, exist_ok=True)

for idx, chunk in enumerate(chunks, start=1):
	filename = os.path.join(out_dir, f'node_data_part_{idx}.json')
	data = []

	len_chunk = len(chunk)
	for _, concept in enumerate(chunk):
		entry = dictionary_entry(concept)
		data.append(entry)
		print(f'{concept} is done! ({_}/{len_chunk}) ({round(_*100/len_chunk, 3)}%)')
		time.sleep(1)  # Optional: Delay to prevent overwhelming the API

	# Save the chunk to a JSON file
	with open(filename, 'w') as f:
		json.dump(data, f, indent=4)
