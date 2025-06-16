#This script searches for papers related to ferroelectricity and saves the results in JSON format.
#VOSviewer while reding shows Error while reading Semantic Scholar JSON file 'papers.json: A JSONObject text must begin with 'f' at 1 [character 2 line 1]
#In a file response_to_json_2.py I tried to avoid this errorby wrapping the data in a dictionary

from semanticscholar import SemanticScholar
import json

sch = SemanticScholar()
response = sch.search_paper(query='ferroelectricity | ferroelectric materials', bulk=True)

with open("test.json", "w", encoding="utf-8") as f:
    json.dump([paper._data if hasattr(paper, "_data") else vars(paper) for paper in response], f, ensure_ascii=False, indent=2)