import gensim


documents = ["Apple is releasing a new product", 
             "Amazon sells many things",
             "Microsoft announces Nokia acquisition"]

stoplist = []

texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)

print lda.print_topics(5)
