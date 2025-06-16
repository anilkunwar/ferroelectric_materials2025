#This script fetches papers related to ferroelectricity and saves first butch of DOIs to a text file from request.
#In VOSviewer information from DOI file has limited application, no keywords co-occurence for example

from semanticscholar import SemanticScholar
sch = SemanticScholar()
response = sch.search_paper(query='ferroelectricity | ferroelectric materials', bulk=True)

# Pobierz pierwsze 1000 rekordów i zamień na dict za pomocą vars()
papers = [vars(paper) for paper in response[:1000]]

# Zapisz tylko DOI do pliku tekstowego
with open("first_batch_dois.txt", "w", encoding="utf-8") as txtfile:
    for paper in papers:
        data = paper.get("_data", paper)
        doi = ""
        if "externalIds" in data and isinstance(data["externalIds"], dict):
            doi = data["externalIds"].get("DOI", "")
        if doi:
            txtfile.write(doi + "\n")

print("Zapisano plik first_batch_dois.txt")