library(tidytext)
library(dplyr)
library(stringr)

## Creating the set of question tags
que=  c("how", "what", "which", "why", "whom", "when", "where", "whose", "who")
## Providing the values to the words
word = c(0,0,0,0,0,0,0,0,0)
## creating data frame using dplyr
match_pair = data_frame(word = que, val = word)


# extracting sample data to to check the code, this part will not be their in final code
#yy = train[c(1:38),]
yy20 = questions_Train[c(400001:404290),]
# creating data frame structure for output 
datagen = function () {
qf = data.frame(how1 = as.numeric(), what1 = as.numeric(), 
                which1 = as.numeric(), why1 = as.numeric(),
                whome1 = as.numeric(), when1 = as.numeric(),
                where1 = as.numeric(), whose1 = as.numeric(),
                who1 = as.numeric(), 
                how2 = as.numeric(), what2 = as.numeric(), 
                which2 = as.numeric(), why2 = as.numeric(),
                whome2 = as.numeric(), when2 = as.numeric(),
                where2 = as.numeric(), whose2 = as.numeric(),
                who2 = as.numeric())
return (qf)

}

# Performing the extraction and count of question tags words 
datafeed = function (yy20){
len= length(yy20$id)
for (i in c(1:len)) {
  
  # Feeding row by row data to 'y' variable
  y = yy20[i,]
  
  # storing questions of a single row in vector format
  question1  = as.vector(y$question1)
  question2  = as.vector(y$question2)
  
  # making tidy data frame out of the complete data set 
  q1 <- data_frame(text = question1)
  q2 <- data_frame(text = question2)
  
  # For question 1, creating token of words, where each row will have a single word for that 
  ## particular row of record
  q1 = q1 %>%
    unnest_tokens(word, text)
  ### filtering question tags word only
  q1 = q1 %>% inner_join(match_pair)
  ### counting the tags 
  q1  = q1 %>% dplyr::count(word)
  # mapping the counts in question to the complete set of tags
  q1 = match_pair %>% left_join(y = q1,by = "word" )
  q1 = q1[,c(1,3)]
  ## Replacing NA with 0 and transposing the data set
  q1 = apply(data.frame(q1 %>% tidyr::spread(word, n)), MARGIN =  c(1,2) ,
             function(x) ifelse(is.na(x),0,x))
  
  # Repeating the complete exercise for question2
  q2 = q2 %>%
    unnest_tokens(word, text)
  q2 = q2 %>% inner_join(match_pair)
  q2  = q2 %>% dplyr::count(word)
  q2 = match_pair %>% left_join(y = q2,by = "word" )
  q2 = q2[,c(1,3)]
  q2 = apply(data.frame(q2 %>% tidyr::spread(word, n)), MARGIN =  c(1,2) ,
             function(x) ifelse(is.na(x),0,x))
  
  
  # combing the results from question1 and 2 and appending it the set of records 
  qf  = rbind(qf, as.data.frame(cbind(q1,q2))) }
  qf = as.data.frame(qf)
  return(qf)
}

# calling function 
qf = datagen()
#Performing the extraction on subset of the data. In final code, we will replace the subsetting with complete dataset
yy20 = questions_Train[c(400001:404290),]
qf = datafeed(yy20)
View(qf)
#renaming columns 
colnames(qf)  = as.vector(colnames(datagen())) 

# Combining the columns with existing dataset
yy20  = data.frame(cbind(yy20,qf))

# Combining the individual rows of extracted features 
sentiment_features = rbind(yy,yy1,yy2,yy3,yy4,yy5,yy6,yy7,yy8,yy9,yy10,yy11,yy12,yy13,yy14,yy15,yy16,yy17,yy18,yy19,yy20)
write.csv(sentiment_features,"C:/Users/Vinodh Mohan/Documents/MSBAPM/Data Analytics using R/Quora_Question_Pairs_Duplicate/SentimentFeatures.csv")
