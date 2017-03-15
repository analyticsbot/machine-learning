import re

##def words_and_char_bigrams(text):
##   words = text.split(';')
##   print words
##   for w in words:
##       yield w
####       for i in range(len(w) - 2):
####           yield w[i:i+2]
##
##print words_and_char_bigrams('Bad Boy goes')


from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import re
x = ['this is a; foo bar', 'you are a foo; bar black sheep']
##def words_and_char_bigrams(text):
##    words = re.findall(r'\w{3,}', text)
##    for w in words:
##        yield w
##        for i in range(len(w) - 2):
##            yield w[i:i+2]
##            
##v = CountVectorizer(analyzer=words_and_char_bigrams)
##pprint(v.fit(x).vocabulary_)


def words_and_char_bigrams(text):
    words = text.split(';')
    for w in words:
        yield w
        
            
v = CountVectorizer(analyzer=words_and_char_bigrams)
pprint(v.fit(x).vocabulary_)
