setwd("C:/MSBAPM_courses/Sem2/R/DataVaders")
list.files()
library(data.table)
library(randomForest)
library(class)
library(e1071)
library(rpart)
library(caret)
#Reading the Train dataset
#merged_data = fread("merged_data.csv",header = TRUE,stringsAsFactors = FALSE,select=2:103)
library(data.table) # has fread
#1st column irrelevant
features_vinoth <- fread("FuzzyFeatures.csv",header = TRUE,stringsAsFactors = FALSE,select=2:11)
#1st columns irrelevant
#features_piyush <- fread("SentimentFeatures.csv",header = TRUE,stringsAsFactors = FALSE,select=8:25)
#1st  7columns irrelevant
features_ravi1  <- fread("train_features2.csv",header = TRUE,stringsAsFactors = FALSE,select=8:22)
#1st 7 columns irrelevant
features_ravi2 <- fread("train_features3.csv",header = TRUE,stringsAsFactors = FALSE,select=8:9)
#1st 7 columns irrelevant
features_ravi3 <- fread("train_features4.csv",header = TRUE,stringsAsFactors = FALSE,select=8:9)
#1st 7 columns irrelevant
features_ravi4 <- fread("train_features5.csv",header = TRUE,stringsAsFactors = FALSE,select=7)
#1st 6 colums irrelevant
features_ravi5 <- fread("train_features6.csv",header = TRUE,stringsAsFactors = FALSE,select=7)
#reading Y column
is_duplicate <- fread("SentimentFeatures.csv",header = TRUE,stringsAsFactors = FALSE,select=7)
merged_data <- cbind(features_vinoth,features_ravi1,
                     features_ravi2,features_ravi3,features_ravi4,features_ravi5,
                     is_duplicate)

merged_data<-data.frame(merged_data)
#merged_data<-merged_data[,-1]#remove unnecessary cols
merged_data<-merged_data[,-c(13:21)]#remove unnecessary cols

#split the train and test data
test_train_split = function(frac, merged_data){
  
  classes = merged_data$is_duplicate
  set.seed(203)
  trainingRows <- createDataPartition(classes,p = frac,list= FALSE)
  return(trainingRows)
}

index = test_train_split(0.020, merged_data)

#Creating test and training set
train_data = merged_data[index ,]
test_data = merged_data[-index ,]

#
#Null Value treatment
train_data = as.data.frame(apply(train_data, MARGIN = c(1,2), function(x) ifelse(is.na(x),0,x)))
test_data = as.data.frame(apply(test_data, MARGIN = c(1,2), function(x) ifelse(is.na(x),0,x)))


# modelling
y_pos<-23 #indicates position of target variable in dataframe
#Modelling for Svm linear classifier

#library(e1071)
svm_linear_classifier= svm(formula = is_duplicate~., data=train_data,type='C-classification',kernel='linear')
#test_data <- test_data[complete.cases(test_data), ]
y_pred_svm_linear= predict(svm_linear_classifier,type="response",newdata=test_data[,-y_pos],
                           nan.rm=TRUE,Inf.rm=TRUE,na.rm=TRUE)

#result
confusionMatrix(test_data[,y_pos],y_pred_svm_linear)

#Modelling for Svm Radial Classifier 
svm_radial_classifier= svm(formula = is_duplicate~.,data=train_data, type='C-classification',kernel='radial')
y_pred_svm_radial= predict(svm_radial_classifier,type="response",newdata=test_data[,-y_pos],
                           nan.rm=TRUE,Inf.rm=TRUE,na.rm=TRUE)

#Result Statistics
confusionMatrix(test_data[,y_pos],y_pred_svm_radial)
test_data$y_pred_svm_radial=y_pred_svm_radial
test_data$y_pred_svm_linear=y_pred_svm_linear

#writing predicted data
write.csv(merged_data,'svm_data.csv')
write.csv(test_data,'test_data')
