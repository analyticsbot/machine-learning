import csv

#f = open('BingHackathonTrainingData.txt', 'r')
f = open('BingHackathonTestData.txt', 'r')

data = f.read().split('\n')[:-1]

#o = open('BingHackathonTrainingData.csv', 'wb')
o = open('BingHackathonTestData.csv', 'wb')
writer = csv.writer(o)
writer.writerow(['record_id', 'topic_id', 'publication_year',\
                 'authors', 'title', 'summary'])

for line in data:
    line = line.strip()
    line = line.split('\t')
    writer.writerow(line)

f.close()
o.close()
