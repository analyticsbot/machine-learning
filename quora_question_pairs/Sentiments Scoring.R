# Install the package "syuzhet" that will be used for sentiment scoring feature
# devtools::install_github("mjockers/syuzhet")


# require(rJava)
# install.packages("syuzhet")

## Loading syuzhet package 
library(syuzhet)

#Uncomment the run the line before if the machine throughs
# Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre7')

#copying the data set to the a new object to make sure the original object remains intact
y = train 

# Making module of the exercise 
senticheck = function(y){
  
  # creating empty data frame 

df_sent = data.frame(ques1_sent_pos = as.numeric() , ques1_sent_neg = as.numeric(),
                     ques2_sent_pos  = as.numeric() , ques2_sent_neg = as.numeric())

## Looping through each question
for (i in (1 : length(y$question1))){
  
  #making vecotor of questions
  ques1 = as.vector(y$question1[i])
  ques2 = as.vector(y$question2[i])
  
  #Tokenization:
  
  ques1 = get_tokens(ques1)
  ques2 = get_tokens(ques2)
  
  #finding sentiment of each word of both the questions 
  ques1_sent = get_sentiment(ques1 ,  method="bing")
  ques2_sent = get_sentiment(ques2 ,  method="bing")
  
  #storing positve and negative sentiment separately
  ques1_sent_pos = sum( ifelse(ques1_sent > 0 ,1,0))
  ques1_sent_neg = sum( ifelse (ques1_sent < 0, 1,0 ))
  
  ques2_sent_pos = sum( ifelse(ques2_sent > 0 ,1,0))
  ques2_sent_neg = sum( ifelse (ques2_sent < 0, 1,0 ))
  
  # making a one row data frame 
  c2 = as.data.frame(cbind(ques1_sent_pos, ques1_sent_neg,
                          ques2_sent_pos , ques2_sent_neg))
  #appending it to main data frame
  df_sent = rbind(df_sent, c2)
  
}
  # jOning the with rest of the data
  sentdata = cbind(y, df_sent)
  sentdata['question1_sentiment'] = ques1_sent_pos + ques1_sent_neg
    sentdata['question2_sentiment'] = ques2_sent_pos + ques2_sent_neg
  return(sentdata)
}


# Running the function 
xs = senticheck(y)

#writing the file to local machine at current working directory 
write.csv(xs , "part1.csv" )
