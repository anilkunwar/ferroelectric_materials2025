#This script searches for papers related to ferroelectricity and saves the results in JSON format.
#VOSviewer while reding shows Error while reading Semantic Scholar JSON file 'papers.json': Invalid JSON data format. Maybe because VOSviewer does not expect a dictionary with a "paper" key.
#I also noticed that if i do search "keywords" in test2.json i only get less than 100 results while thers over 1000 papers in general and that it's only a part of the value which is linked to the key "abstarct"

from semanticscholar import SemanticScholar
import json

sch = SemanticScholar()
response = sch.search_paper(query='ferroelectricity | ferroelectric materials', bulk=True)

papers = [paper._data if hasattr(paper, "_data") else vars(paper) for paper in response]

with open("test2.json", "w", encoding="utf-8") as f:
    json.dump({"papers": papers}, f, ensure_ascii=False, indent=2)