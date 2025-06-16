from semanticscholar import SemanticScholar
sch = SemanticScholar()
response = sch.search_paper(query='ferroelectricity | ferroelectric materials', bulk=True)

print(response)
print(type(response))
print(response[0])
print(type(response[0])) 