import os
import csv
import pandas as pd


os.makedirs("./nodes", exist_ok=True)

df = pd.read_csv("concepts_taxonomy.csv")
df = df[df['parent_ids'].notna()]

root_domains = {
	"https://openalex.org/c17744445": "Political_Science",
	"https://openalex.org/c138885662": "Philosophy",
	"https://openalex.org/c162324750": "Economics",
	"https://openalex.org/c144133560": "Business",
	"https://openalex.org/c15744967": "Psychology",
	"https://openalex.org/c33923547": "Mathematics",
	"https://openalex.org/c71924100": "Medicine",
	"https://openalex.org/c86803240": "Biology",
	"https://openalex.org/c41008148": "Computer_Science",
	"https://openalex.org/c127313418": "Geology",
	"https://openalex.org/c185592680": "Chemistry",
	"https://openalex.org/c142362112": "Art",
	"https://openalex.org/c144024400": "Sociology",
	"https://openalex.org/c127413603": "Engineering",
	"https://openalex.org/c205649164": "Geography",
	"https://openalex.org/c95457728": "History",
	"https://openalex.org/c192562407": "Materials_Science",
	"https://openalex.org/c121332964": "Physics",
	"https://openalex.org/c39432304": "Environmental_Science"
}

for domain, name in root_domains.items():
    list_of_parents = [domain]
    for i in df.values:
        for parent in list_of_parents:
            if parent in i[6].lower():
                list_of_parents.append(i[0])
                break

    with open(f'./nodes/{name}.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for item in list_of_parents:
            writer.writerow(["C" + item[22:]])

    print(f"{name} with {len(list_of_parents)} nodes is done!")
