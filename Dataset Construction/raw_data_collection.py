from datetime import datetime
from threading import Thread
import requests as rq
from time import sleep
import json


def get_data(year, cursor):
  count = 0
  l = []
  while cursor:
    url = f"https://api.openalex.org/works?filter=type:types/article,publication_year:{year},language:languages/en&per_page=200&select=concepts&cursor={cursor}"
    while True:
        try:
            res = rq.get(url, timeout=20)
            break
        except Exception as e:
            sleep(2)
            print(e, "\ntrying again\n", sep="\n")
    sleep(1)
    if res.status_code != 200:
      raise Exception(res.status_code)
    d = res.json()
    for result in d["results"]:
      ids = []
      for concept in result["concepts"]:
        ids.append(concept["id"][21:])
      l.append(ids)
    cursor = d["meta"]["next_cursor"]
    count += 1
    if count % 100 == 0:
        print(f"-- {year}, count={count}")
    if count % 2000 == 0:
      file_name = f"{year}_{str(datetime.now())}_{cursor}.json"
      with open("./"+file_name, "wt") as f:
        json.dump(l, f, separators=(',', ':'))
      l = []
  file_name = f"{year}_final.json"
  with open("./"+file_name, "wt") as f:
    json.dump(l, f, separators=(',', ':'))

start_year = 2010
end_year = 2020

years = list(range(start_year, end_year+1))
threads = []

for i, year in enumerate(years):
  new_t = Thread(target=get_data, args=[year, "*"])
  threads.append(new_t)
  print(f"starting {year} thread")
  new_t.start()
  if (i+1) % 10 == 0:
    for t in threads:
      t.join()
    threads = []

