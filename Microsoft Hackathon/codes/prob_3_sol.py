import csv

f = open('prob_3.csv', 'r')
data = f.read().split('\n')[1:-1]

o= open('prob_3_sol.csv', 'w')
writer = csv.writer(o)

for line in data:
    line = line.strip()
    line = line.split(',')
    title = line[3].split()[1:4]
    year = line[1] + str(1)
    summary = line[4].split()[1:100]
    authors = line[2].split(';')[0]

    writer.writerow([year, authors, title, summary])

o.close()
f.close()
    
