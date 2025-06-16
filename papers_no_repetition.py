#This script fetches papers related to ferroelectricity and saves all DOIs to a text file without duplication.
#In VOSviewer information from DOI file has limited application, no keywords co-occurence for example

from semanticscholar import SemanticScholar

sch = SemanticScholar()
query = 'ferroelectricity | ferroelectric materials'
batch_size = 100
all_dois = set()
total_fetched = 0

with open("all_dois.txt", "w", encoding="utf-8") as txtfile:
    next_cursor = None
    while True:
        if next_cursor:
            response = sch.search_paper(query=query, limit=batch_size, next=next_cursor, bulk=True)
        else:
            response = sch.search_paper(query=query, limit=batch_size, bulk=True)
        papers = [vars(paper) for paper in response]
        if not papers:
            break  # No more results

        new_dois = 0
        for paper in papers:
            data = paper.get("_data", paper)
            doi = ""
            if "externalIds" in data and isinstance(data["externalIds"], dict):
                doi = data["externalIds"].get("DOI", "")
            if doi and doi not in all_dois:
                txtfile.write(doi + "\n")
                all_dois.add(doi)
                new_dois += 1

        total_fetched += len(papers)
        print(f"Fetched {len(papers)} papers, {new_dois} new DOIs, total DOIs: {len(all_dois)}")

        # Try to get the next cursor for pagination
        if hasattr(response, 'next'):
            next_cursor = response.next
        elif hasattr(response, 'next_cursor'):
            next_cursor = response.next_cursor
        else:
            break  # No pagination info, stop

        if not next_cursor or len(papers) < batch_size:
            break  # Last page reached

print(f"Zapisano plik all_dois.txt z {len(all_dois)} unikalnymi DOI")