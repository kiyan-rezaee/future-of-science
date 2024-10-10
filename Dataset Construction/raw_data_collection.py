from datetime import datetime
from threading import Thread
import requests as rq
from time import sleep
import json

################################################ My Function   : get_data  #
# Description    : Fetches article concept IDs from OpenAlex API for a given year  #
#                  and saves them to JSON files.                                 #
# Input         : year (int) - The year for which to retrieve article concepts         #
#                  cursor (str) - Optional cursor for pagination (from previous API call) #
# Outputs       : None (saves data to JSON files)                                 #
################################################

def get_data(year, cursor):
    """
    Fetches article concept IDs from OpenAlex API for a given year and saves them to JSON files.

    Args:
        year (int): The year for which to retrieve article concepts.
        cursor (str, optional): Optional cursor for pagination (from previous API call). Defaults to "*".
    """

    count = 0
    l = []  # List to store concept IDs for the current year

    while cursor:
        # Construct the API URL with pagination cursor
        url = f"https://api.openalex.org/works?filter=type:types/article,publication_year:{year},language:languages/en&per_page=200&select=concepts&cursor={cursor}"

        while True:
            try:
                # Make a request to the OpenAlex API with timeout
                res = rq.get(url, timeout=60)
                if res.status_code == 200:
                    # Success! Break the retry loop
                    break
                else:
                    # Handle non-200 status codes (e.g., rate limiting)
                    sleep(2)
                    print(f"status code: {res.status_code}", "\ntrying again\n", sep="\n")
            except Exception as e:
                # Handle any exceptions that might occur during the request
                sleep(2)
                print(e, "\ntrying again\n", sep="\n")

        sleep(1)  # Small delay between requests

        # Parse the JSON response
        d = res.json()

        for result in d["results"]:
            # Extract concept IDs from each article result
            ids = []
            for concept in result["concepts"]:
                ids.append(concept["id"][21:])  # Extract ID substring
            l.append(ids)

        # Update cursor for pagination (if available)
        cursor = d["meta"].get("next_cursor", None)

        count += 1

        # Print progress every 100 requests
        if count % 100 == 0:
            print(f"-- {year}, count={count}")

        # Save data to JSON files every 2000 requests
        if count % 2000 == 0:
            file_name = f"{year}_{str(datetime.now())}_{cursor}.json"
            with open("./" + file_name, "wt") as f:
                json.dump(l, f, separators=(',', ':'))  # Dump data with specific separators
            l = []  # Reset list for next 2000 requests

    # Final save after processing all results
    file_name = f"{year}_final.json"
    with open("./" + file_name, "wt") as f:
        json.dump(l, f, separators=(',', ':'))



start_year = 2010
end_year = 2020

threads = []
years = list(range(start_year, end_year+1))
cursors = {year: "*" for year in years}

# override any cursor you want
# cursors[2010] = abcd...

for i, year in enumerate(years):
  new_t = Thread(target=get_data, args=[year, cursors[year]])
  threads.append(new_t)
  print(f"starting {year} thread")
  new_t.start()
  if (i+1) % 10 == 0:
    for t in threads:
      t.join()
    threads = []

