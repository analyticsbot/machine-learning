# Extracting the Features from the Question Pairs

# Extracting Fuzzy Features
#install.packages("fuzzywuzzyR")
library(fuzzywuzzyR)

init = FuzzMatcher$new() #initializing the Fuzz Matcher class

#Extracting Fuzzy Feature 1 - QRatio
Qratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$QRATIO(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i],force_ascii = TRUE)
  Qratio[i] = ratio
}

#Extracting Fuzzy Feature 2 - WRatio
Wratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$WRATIO(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i],force_ascii = TRUE)
  Wratio[i] = ratio
}

#Extracting Fuzzy Feature 3 - PartialRatio
Part_ratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$Partial_ratio(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i])
  Part_ratio[i] = ratio
}

#Extracting Fuzzy Feature 4 - PartialTokenSetRatio
Part_tok_set_ratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$Partial_token_set_ratio(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i],force_ascii = TRUE,full_process = TRUE)
  Part_tok_set_ratio[i] = ratio
}

#Extracting Fuzzy Feature 5 - PartialTokenSortRatio
Part_tok_sort_ratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$Partial_token_sort_ratio(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i],force_ascii = TRUE,full_process = TRUE)
  Part_tok_sort_ratio[i] = ratio
}

#Extracting Fuzzy Feature 6 - TokenSetRatio
Tok_set_ratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$Token_set_ratio(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i],force_ascii = TRUE,full_process = TRUE)
  Tok_set_ratio[i] = ratio
}

#Extracting Fuzzy Feature 7 - TokenSortRatio
Tok_sort_ratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$Token_sort_ratio(string1 = questions_Train$question1[i],
                               string2 = questions_Train$question2[i],force_ascii = TRUE,full_process = TRUE)
  Tok_sort_ratio[i] = ratio
}

#Extracting Fuzzy Feature 8 - UQRatio
UQratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$UQRATIO(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i])
  UQratio[i] = ratio
}

#Extracting Fuzzy Feature 9 - UWRatio
UWratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$UWRATIO(string1 = questions_Train$question1[i],
                      string2 = questions_Train$question2[i])
  UWratio[i] = ratio
}

#Extracting Fuzzy Feature 10 - Ratio
Ratio = list()
for(i in 1:length(questions_Train$id)){
  ratio = init$Ratio(string1 = questions_Train$question1[i],
                       string2 = questions_Train$question2[i])
  Ratio[i] = ratio
}

#Creating a data frame after combining all these features
Fuzzy_Features = cbind(Qratio,Wratio,Part_ratio,
                                     Part_tok_set_ratio,Part_tok_sort_ratio,
                                     Tok_set_ratio,Tok_sort_ratio,
                                     UQratio,UWratio,Ratio)

Fuzzy_Features <- data.frame(matrix(unlist(Fuzzy_Features),
                                    nrow=length(Fuzzy_Features)/10,
                                    byrow=T),
                             stringsAsFactors=FALSE)

names(Fuzzy_Features) = c('Qratio','Wratio','Part_ratio',
                          'Part_tok_set_ratio','Part_tok_sort_ratio',
                          'Tok_set_ratio','Tok_sort_ratio',
                          'UQratio','UWratio','Ratio')

#Writing the dataframe to csv
write.csv(Fuzzy_Features,"C:/Users/Vinodh Mohan/Documents/MSBAPM/Data Analytics using R/Quora_Question_Pairs_Duplicate/FuzzyFeatures.csv")
