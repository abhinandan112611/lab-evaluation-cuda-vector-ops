import csv

rows = list(csv.DictReader(open("results/results.csv")))
print("rows:", len(rows))
print("errors>0:", sum(int(x["errors"]) > 0 for x in rows))