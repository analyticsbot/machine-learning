import gensim, csv

f = open('BingHackathonTrainingData.txt', 'r')
data = f.read().split('\n')[1:-1]

documents = []

##Year of publication: up to you.
##Authors: only from authors in training data; not more than 3 authors; semicolon-separated.
##Title: not more than 10 words (token IDs), from the existing vocabulary
##Summary: only from the set of sentences in all the training data summaries; not more than 8 sentences. A sentence is a sequence of token IDs, space separated, and followed by a period.

#title
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

# get words in a topic
word_topics = []
count = 0 
for topic in topics:
    topic_split = topic.split('+')
    for t_split in topic_split:
        words = [w.strip() for w in t_split.split('*')]
        if count == 0:
            print words
        count +=1
        if words[1] not in word_topics:
            word_topics.append(words[1])

    break


    

    
