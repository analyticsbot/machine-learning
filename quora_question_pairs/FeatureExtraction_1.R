###################################################################################
# The problem statement - Quora Duplicate Question Prediction
# This R code extracts feature from the question pairs which will then be fed
# to a machine learning model for predictions
###################################################################################

## to begin, it is required to install these packages

install.packages(c("tm", "stringr", "vegan", "RSentiment", 'SnowballC', "Stringdist", "NLP", "openNLP"))

## next, set the JAVA heap to 8GB. This was done due to one of the errors that occurred without running
## this. by default, the JAVA heap is 512MB
options(java.parameters = "-Xmx8000m")

## load the required packages
library(tm)
library(stringr)
library(vegan)
library(RSentiment)
library(stringdist)
library(NLP)
library(openNLP)

## read the training dataset
train <- read.csv("~/jeremy/quora/train.csv", stringsAsFactors = F)

## dimensions of the dataset
dim(train)

## next step is to extract the features from this dataset of question1 and queston2

## FEATURE 1 - LENGTH OF QUESTION1 AND QUESTION2, THE DIFFERENCE AND DIVISION
## find the length of question. this includes spaces as well
lengthQuestion <- function(question){
  return(nchar(question))
}

## ADD FEATURES TO THE TRAINING DATASET
## lenQ1 is the length of question1, lenQ2 is the length of question2
## lenDiff is lenQ1-lenQ2, lenDivide is lenQ1/lenQ2
## the idea is two questions can't be similar if they have really different lengths
train[c("lenQ1")] <- lapply(train[c("question1")], lengthQuestion)
train[c("lenQ2")] <- lapply(train[c("question2")], lengthQuestion)
train[c("lenDiff")] <- train[c("lenQ1")]-train[c("lenQ2")]
train[c("lenDivide")] <- train[c("lenQ1")]/train[c("lenQ1")]

## FEATURE 2 - NUMBER OF WORDS IN QUESTION1 AND QUESTION2, THE DIFFERENCE AND DIVISION
## number of words in a question
numWords <- function(question){
  return (sapply(strsplit(question, " "), length))
}

## ADD FEATURES TO THE TRAINING DATASET
## numWordsQ1 is the number of words in question1, numWordsQ2 is the number of words in question2
## numWordsDiff is numWordsQ1-numWordsQ2 and numWordsDivide is numWordsQ1/numWordsQ2
train[c("numWordsQ1")] <- lapply(train[c("question1")], numWords)
train[c("numWordsQ2")] <- lapply(train[c("question2")], numWords)
train[c("numWordsDiff")] <- train[c("numWordsQ1")]-train[c("numWordsQ2")]
train[c("numWordsDivide")] <- train[c("numWordsQ1")]/train[c("numWordsQ2")]


## FEATURE 3 - NUMBER OF CHARACTERS IN QUESTION1 AND QUESTION2, THE DIFFERENCE AND DIVISION
## THIS DOES NOT COUNT SPACES
numChars <- function(question){
  return(nchar(question) - length(gregexpr(" ", question)[[1]]))
}

## ADD FEATURES TO THE TRAINING DATASET
## numCharsQ1 is the number of characters in question1, numCharsQ2 is the number of characters in question2
## numCharsDiff is numCharsQ1-numCharsQ2 and numCharsDivide is numCharsQ1/numCharsQ2
train[c("numCharsQ1")] <- lapply(train[c("question1")], numWords)
train[c("numCharsQ2")] <- lapply(train[c("question2")], numWords)
train[c("numCharsDiff")] <- train[c("numCharsQ1")]-train[c("numCharsQ2")]
train[c("numCharsDivide")] <- train[c("numCharsQ1")]/train[c("numCharsQ2")]

## FEATURE 4 - NUMBER OF COMMON WORDS IN QUESTION1 AND QUESTION2
## number of common words between two questions
numCommonWords <- function(question1, question2){
  split1 <- unlist(strsplit(question1, split=" "))
  split2 <- unlist(strsplit(question2, split=" "))
  return(length(intersect(split1, split2)))
}

## ADD FEATURES TO THE TRAINING DATASET
## numCommonWords is the number of common words
train[c("numCommonWords")] <- mapply(numCommonWords, train$question1, train$question2)


## FEATURE 5 - NUMBER OF COMMON WORDS IN QUESTION1 AND QUESTION2 NORMALIZED (DIVIDED BY LENGTH OF QUESTION1 AND QUESTION2)
## number of common words normalized
numCommonWordsNormalized <- function(question1, question2) {
  return(numCommonWords(question1, question2)/(numWords(question1)*numWords(question2)))
}

## ADD FEATURES TO THE TRAINING DATASET
train[c("numCommonWordsNormalized")] <- mapply(numCommonWordsNormalized, train$question1, train$question2)


## FEATURE 6 - NUMBER OF CHARACTERS IN QUESTION1 AND QUESTION2  
## (DIVIDED BY LENGTH OF QUESTION1 AND QUESTION2)
## number of characters divided by length of question
numCharByWord <- function(question){
  return(numChars(question)/lengthQuestion(question))
}

## ADD FEATURES TO THE TRAINING DATASET
## numCharByWordQ1 is the number of characters per word in question1
## numCharByWordQ2 is the number of characters per word in question2
## numCharByWordDiff is numCharByWordQ1 - numCharByWordQ2
## numCharByWordDivide is numCharByWordQ1/numCharByWordQ2
train[c("numCharByWordQ1")] <- lapply(train[c("question1")], numCharByWord)
train[c("numCharByWordQ2")] <- lapply(train[c("question2")], numCharByWord)
train[c("numCharByWordDiff")] <- train[c("numCharByWordQ1")]-train[c("numCharByWordQ2")]
train[c("numCharByWordDivide")] <- train[c("numCharByWordQ1")]/train[c("numCharByWordQ2")]


## FEATURE 7 - NUMBER OF CHARACTERS IN QUESTION1 AND QUESTION2  
## (DIVIDED BY LENGTH OF QUESTION1 AND QUESTION2)
## gives how many a, b, c, ... z are there in question1 and question2
## then take the difference between the two questions
## the idea is similar sentences might have similar distribution of characters
charTermFrequency <- function(question){
  df <- data.frame(matrix(ncol = length(letters), nrow = 0))
  x <- unlist(list(letters))
  colnames(df) <- x
  for (i in letters){
    df[1, i] = str_count(question, i)
  }
  return(df)
}
## ADD FEATURES TO THE TRAINING DATASET
## x contains the characters a-z in question1
## y contains the characters a-z in question2
## z contains the difference of characters a-z between question1 and question2
x = t(sapply(train$question1,charTermFrequency, USE.NAMES=F))
y = t(sapply(train$question2,charTermFrequency, USE.NAMES=F))
z = sapply(as.data.frame(x), as.integer) - sapply(as.data.frame(y), as.integer)

train = cbind(train, z)


## FEATURE 7 - COSINE SIMILARITY BETWEEN QUESTION1 AND QUESTION2  
## returns the cosine similarity between two sentences
cosineSimilarity <- function(question1, question2){
  A = charTermFrequency(question1)
  B = charTermFrequency(question2)
  return(1- sum(A*B)/sqrt(sum(A^2)*sum(B^2)))
}

## ADD FEATURES TO THE TRAINING DATASET
train[c("cosineSimilarity")] <- mapply(cosineSimilarity, train$question1, train$question2)

## FEATURE 8 - MULTIPLE SIMILARITY USING DIST PACKAGE BETWEEN QUESTION1 AND QUESTION2  
# "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski" 
distSimilarity <- function(question1, question2, type="euclidean"){
  A = charTermFrequency(question1)
  B = charTermFrequency(question2)
  similarity = dist(rbind(A, B), method = type)[[1]]
  return(similarity)
}
## ADD FEATURES TO THE TRAINING DATASET
## ONLY ADDED COLUMNS IN THE FINAL DATASET THAT RAN WITHOUT ANY ERROR
train[c("binarySimilarity")] <- mapply(distSimilarity, train$question1, train$question2, "binary")
train[c("maximumSimilarity")] <- mapply(distSimilarity, train$question1, train$question2, "maximum")
train[c("minkowskiSimilarity")] <- mapply(distSimilarity, train$question1, train$question2, "minkowski")

## FEATURE 9 - MULTIPLE SIMILARITY USING VEGDIST PACKAGE BETWEEN QUESTION1 AND QUESTION2  
## vegan package
#"manhattan", "euclidean", "canberra", "bray", "kulczynski", "jaccard",
# "gower", "altGower", "morisita", "horn", "mountford", "raup" , 
#"binomial" or "chao"
veganSimilarity <- function(question1, question2, type="euclidean"){
  A = charTermFrequency(question1)
  B = charTermFrequency(question2)
  similarity = vegdist(rbind(A, B), method = type)[[1]]
  return(similarity)
}

## ADD FEATURES TO THE TRAINING DATASET
## ONLY ADDED COLUMNS IN THE FINAL DATASET THAT RAN WITHOUT ANY ERROR
train[c("manhattanSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "manhattan")
train[c("euclideanSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "euclidean")
train[c("canberraSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "canberra")
train[c("braySimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "bray")
train[c("kulczynskiSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "kulczynski")
train[c("jaccardSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "jaccard")
train[c("gowerSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "gower")
train[c("altGowerSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "altGower")
train[c("morisitaSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "morisita")
train[c("hornSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "horn")
train[c("mountfordSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "mountford")
train[c("raupSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "raup")
train[c("binomialSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "binomial")
train[c("chaoSimilarity")] <- mapply(veganSimilarity, train$question1, train$question2, "chao")


## FEATURE 10 - MULTIPLE SIMILARITY USING STRINGDIST PACKAGE BETWEEN QUESTION1 AND QUESTION2  
## some more distances
# "osa", "lv", "dl", "hamming", "lcs", "qgram",
#"cosine", "jaccard", "jw", "soundex"
moreDistSimilarity<- function(question1, question2, type){
  stringdist(question1, question1, method = c(type))
}

## ADD FEATURES TO THE TRAINING DATASET
## ONLY ADDED COLUMNS IN THE FINAL DATASET THAT RAN WITHOUT ANY ERROR
train[c("osaSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "osa")
train[c("lvSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "lv")
train[c("dlSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "dl")
train[c("hammingSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "hamming")
train[c("lcsSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "lcs")
train[c("qgramSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "qgram")
train[c("jwSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "jw")
train[c("soundexSimilarity")] <- mapply(moreDistSimilarity, train$question1, train$question2, "soundex")


## FEATURE 11 - SENTIMENT OF QUESTION1 AND QUESTION2  
## get sentiment of the question

getSentiment <- function(question){
  return(calculate_score(question)[[1]])
}

## ADD FEATURES TO THE TRAINING DATASET
## sentimentQ1 is sentiment of question1
## sentimentQ2 is sentiment of question2
train[c("sentimentQ1")] <- lapply(train[c("question1")], getSentiment)
train[c("sentimentQ2")] <- lapply(train[c("question2")], getSentiment)

## FEATURE 12 - NUMBER OF INTERROGATIVE WORDS IN QUESTION1 AND QUESTION2  
## AND THEIR DIFFERENCE BETWEEN QUESTION1 AND QUESTION2
## IDEA IS IF ONE SENTENCE BEGINS WITH HOW AND OTHER WITH WHEN, THE
## PROBABILITY IS LOW THESE TWO ARE ASKING THE SAME THING
## interrogative words
## get frequency of interrogative words
interrogativeWords = c("when", "what", "where", "who", "whom", "whose", "why", "which", "how")

getInterrogativeWordCount <- function(question, interrogativeWords){
  df <- data.frame(matrix(ncol = length(interrogativeWords), nrow = 0))
  x <- unlist(list(interrogativeWords))
  colnames(df) <- x
  for (i in interrogativeWords){
    df[1, i] = str_count(question, i)
  }
  return(df)
}

## ADD FEATURES TO THE TRAINING DATASET
## X2 IS THE NUMBER OF OCCURENCES OF INTERROGATIVE WORDS IN QUESTION1
## Y2 IS THE NUMBER OF OCCURENCES OF INTERROGATIVE WORDS IN QUESTION2
## Z2 IS THE DIFFERENCE OF OCCURENCES OF INTERROGATIVE WORDS IN QUESTION1 AND QUESTION2
x2 = t(sapply(train$question1,getInterrogativeWordCount, interrogativeWords, USE.NAMES=F))
y2 = t(sapply(train$question2,getInterrogativeWordCount, interrogativeWords, USE.NAMES=F))

z2 = sapply(as.data.frame(x2), as.integer) - sapply(as.data.frame(y2), as.integer)
train = cbind(train, z2)

## FUNCTION TO PREPROCESS THE DATA
## CONVERT TO LOWER CASE
## REMOVE PUNCTUATION
## REMOVE STOPWORDS IN THE ENGLISH LANGUAGE
## STEM WORDS TO ROOT FORM
preProcessText <- function(question){
  # normalization of the text:
  question <- Corpus(VectorSource(question))
  question <- tm_map(question, tolower) #lowercase
  question <- tm_map(question, removePunctuation, preserve_intra_word_dashes = TRUE) # remove punctuation
  question <- tm_map(question, removeWords, stopwords("english")) # remove stopwords
  question <- tm_map(question, stemDocument) # reduce word forms to stems
  return(as.character(question[[1]]))
}


###############################################################
###############################################################
### THIS FEATURE TOOK TOO MUCH TIME TO RUN AND HENCE IS A SUGGESTION AND
### HAS NOT BEEN CONSIDERED AS A FEATURE FOR THIS PROJECT
### THE IDEA IS POS TAGS MIGHT PLAY A ROLE IN THE DIFFERENCE OF TWO QUESTIONS
## get frequency of POS tags
postTagsTermFrequency <-  function(question) {
  allPOSTags = c("CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", 
                 "MD", "NN", "NNS", "NNP", "NNP", "PDT", "POS", "PRP", "PRP", 
                 "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", 
                 "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB")
  s <- as.String(question)
  word_token_annotator <- Maxent_Word_Token_Annotator()
  a2 <- Annotation(1L, "sentence", 1L, nchar(s))
  a2 <- annotate(s, word_token_annotator, a2)
  a3 <- annotate(s, Maxent_POS_Tag_Annotator(), a2)
  a3w <- a3[a3$type == "word"]
  POStags <- unlist(lapply(a3w$features, `[[`, "POS"))
  o = list(POStags)
  a <- table(o)
  df <- data.frame(matrix(ncol = length(allPOSTags), nrow = 0))
  x <- unlist(allPOSTags)
  colnames(df) <- x
  for (i in allPOSTags){
    
    count = a[names(a)==i]
    if (length(count)==0){
      df[1, i] = 0
    }
    else{
      df[1, i] = count
    }
    
  }
  df[is.na(df)] <- 0
  return(df)
}

train_sample = train[1:100, ]
x1 = t(sapply(train$question1,postTagsTermFrequency, USE.NAMES=F))
y1 = t(sapply(train$question2,postTagsTermFrequency, USE.NAMES=F))

z1 = sapply(as.data.frame(x1), as.integer) - sapply(as.data.frame(y1), as.integer)

train = cbind(train, z1)

