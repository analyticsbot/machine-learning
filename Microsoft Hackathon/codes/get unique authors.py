import csv, operator

f = open('BingHackathonTrainingData.txt', 'r')

data = f.read().split('\n')[1:-1]

# unique authors
o = open('BingHackathonTrainingDataUniqueAuthors.csv', 'wb')
unique_authors = []

writer = csv.writer(o)
writer.writerow(['authors'])

for line in data:
    line = line.strip()
    line = line.split('\t')
    authors = line[3].split(';')

    for author in authors:
        if author not in unique_authors:
            unique_authors.append(author)
            writer.writerow([author])

f.close()
o.close()

# unique title
o = open('BingHackathonTrainingDataUniqueTitle.csv', 'wb')
writer = csv.writer(o)
writer.writerow(['titles'])
unique_titles = []
for line in data:
    line = line.strip()
    line = line.split('\t')
    title = line[4]
    if title not in unique_titles:
        unique_titles.append(title)
        writer.writerow([title])

o.close()

# unique words
o = open('BingHackathonTrainingDataUniqueWords.csv', 'wb')
writer = csv.writer(o)
writer.writerow(['words', 'freq'])

words = {}
for line in data:
    line = line.strip()
    line = line.split('\t')
    word = line[5].split()
    for w in word:
        if w not in words.keys():
            words[w] = 1
        else:
            words[w] += 1

sorted_words = sorted(words.items(), key=operator.itemgetter(1), reverse = True)
for word, freq in sorted_words:
    writer.writerow([word, freq])

o.close()

