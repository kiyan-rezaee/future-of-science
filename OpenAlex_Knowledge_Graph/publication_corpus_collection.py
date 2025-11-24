import os
import json
import requests as rq
from time  import sleep
from threading import Thread
from datetime import datetime


###############################################
# Get Data      : Fetches article data for a range of years using threads and saves JSON files.
# Input         : year (int) - The year for which to fetch data
#               : cursor (str) - The cursor for pagination in the API
# Outputs       : Saves JSON files with concept data for each year
###############################################
def get_data(year, cursor):
    #------------------------------------------
    # Block to Initialize Variables
    # Initializes the count and list for storing concepts
    #------------------------------------------
    # Initialize count and list to store concepts
    count = 0
    l = []

    #------------------------------------------
    # Block to Fetch Data from API
    # Continuously fetches data using pagination with the cursor
    #------------------------------------------
    # Loop while cursor exists to paginate API requests
    while cursor:
        # Construct the URL for the API request
        url = f"https://api.openalex.org/works?filter=type:types/article,publication_year:{year},language:languages/en&per_page=200&select=concepts&cursor={cursor}"

        #------------------------------------------
        # Block to Handle API Request
        # Retries the API request if it fails
        #------------------------------------------
        while True:
            try:
                # Send GET request to the API
                res = rq.get(url, timeout=60)

                # Break loop if request is successful
                if res.status_code == 200:
                    break

                # If status code is not 200, retry after a short delay
                sleep(2)
                print(f"status code: {res.status_code}", "\ntrying again\n", sep="\n")
            except Exception as e:
                # Print the error and retry after a delay
                sleep(2)
                print(e, "\ntrying again\n", sep="\n")

        # Pause between requests to avoid overloading the server
        sleep(1)

        #------------------------------------------
        # Block to Parse API Response
        # Extracts concept IDs from the API response and updates the cursor
        #------------------------------------------
        # Parse the JSON response
        d = res.json()

        # Extract concept IDs from the response
        for result in d["results"]:
            ids = [concept["id"][21:] for concept in result["concepts"]]
            l.append(ids)

        # Update cursor for pagination
        cursor = d["meta"]["next_cursor"]

        #------------------------------------------
        # Block to Monitor Progress
        # Print status updates at regular intervals
        #------------------------------------------
        # Increment the counter and print status after every 100 requests
        count += 1
        if count % 100 == 0:
            print(f"-- {year}, count={count}")

        #------------------------------------------
        # Block to Save Data Periodically
        # Saves partial data to a file after every 2000 requests
        #------------------------------------------
        if count % 2000 == 0:
            # Create a filename based on year and timestamp
            file_name = f"{year}_{str(datetime.now())}_{cursor}.json"

            # Save the list of concepts to a JSON file
            with open("./json_files/" + file_name, "wt") as f:
                json.dump(l, f, separators=(',', ':'))

            # Reset the list after saving
            l = []

    #------------------------------------------
    # Block to Save Final Data
    # Saves the remaining data to a final file
    #------------------------------------------
    # Save any remaining data to a final JSON file
    file_name = f"{year}_final.json"
    with open("./json_files/" + file_name, "wt") as f:
        json.dump(l, f, separators=(',', ':'))


###############################################
# Main          : Main function to start multiple threads for fetching data for different years
# Input         : None
# Outputs       : Starts threads to fetch and save data for a range of years
###############################################
def main():
    #------------------------------------------
    # Block to Set Year Range
    # Defines the range of years for which data is fetched
    #------------------------------------------
    # Define the start and end year for data fetching
    start_year = 1827
    end_year = 2025

    #------------------------------------------
    # Block to Initialize Threads
    # Creates and starts threads for each year
    #------------------------------------------
    # List to hold thread objects
    threads = []

    # Ensure the json_files directory exists
    os.makedirs("./json_files", exist_ok=True)

    # Define the range of years to process
    years = list(range(start_year, end_year + 1))

    # Dictionary to store cursors for each year
    cursors = {year: "*" for year in years}

    #------------------------------------------
    # Block to Override Cursors (if needed)
    # Allows overriding of the cursor for specific years
    #------------------------------------------
    # Override any specific cursor if needed (optional)
    # Example: cursors[2010] = 'abcd...'

    #------------------------------------------
    # Block to Start Threads
    # Creates and starts a new thread for each year
    #------------------------------------------
    # Start a thread for each year
    for i, year in enumerate(years):
        # Create a new thread for the `get_data` function
        new_t = Thread(target=get_data, args=[year, cursors[year]])
        threads.append(new_t)

        # Print message to indicate thread start
        print(f"starting {year} thread")

        # Start the thread
        new_t.start()

        #------------------------------------------
        # Block to Join Threads
        # Waits for a group of threads to finish every 10 threads
        #------------------------------------------
        # Join threads after every 10 threads
        if (i + 1) % 10 == 0:
            for t in threads:
                t.join()
            threads = []

# Call the main function to start the process
if __name__ == "__main__":
    main()
