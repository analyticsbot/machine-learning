import gensim, csv

f = open('BingHackathonTrainingData.txt', 'r')
data = f.read().split('\n')[1:-1]

documents = []

for line in data:
    line = line.strip()
    line = line.split('\t')
    line = line[3]
    documents.append(line)

stoplist = []

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, update_every=1, chunksize=10000, passes=1)

topics = lda.print_topics(num_words = len(dictionary))

for topic in topics:
    topic_split = topic.split('+')
    for t_split in topic_split:
        words = [w.strip() for w in t_split.split('*')]
        #print words

    #print '********'

for topic in topics:
    o = open('word_topic_' + str(topics.index(topic)) + '.csv', 'wb')
    writer = csv.writer(o)
    writer.writerow(['prob', 'word'])

    topic_split = topic.split('+')
    for t_split in topic_split:
            words = [w.strip() for w in t_split.split('*')]
            writer.writerow(words)
    o.close()

# get words in a topic
word_topics = []
for topic in topics:
    t = []
    topic_split = topic.split('+')
    for t_split in topic_split:
            words = [w.strip() for w in t_split.split('*')]
            t.append(words[1])
    word_topics.append(t)

words_only_in_0_ = []
# get words present in topic 0 but not in 1 and 2
for w in word_topics[0]:
    if (w not in word_topics[1]) and (w not in word_topics[2]):
        words_only_in_0_.append(w)

words_only_in_1_ = []
# get words present in topic 1 but not in 0 and 2
for w in word_topics[1]:
    if (w not in word_topics[0]) and (w not in word_topics[2]):
        words_only_in_1_.append(w)

words_only_in_2_ = []
# get words present in topic 2 but not in 1 and 0
for w in word_topics[2]:
    if (w not in word_topics[1]) and (w not in word_topics[0]):
        words_only_in_2_.append(w)


## check what is the topic_id for words that only come in 0
## ideally it should be 0
for line in data:
    line = line.strip()
    line = line.split('\t')
    title = line[3].split()
    for w in words_only_in_0_:
        if w in title:
            print line[1]
    
    

    
