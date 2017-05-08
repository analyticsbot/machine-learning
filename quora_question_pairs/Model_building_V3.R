#Loading library
library(caret)
library(xgboost)
library(readr)
library(Matrix)
library(randomForest)
library(party)
############# Distributing data for train and test:
test_train_split = function(frac, merged_data){
  
  classes = merged_data$is_duplicate
  set.seed(203)
  trainingRows <- createDataPartition(classes,p = frac,list= FALSE)
  return(trainingRows)
}


index = test_train_split(0.005, merged_data)

#Creating test and training set
train_data = merged_data[index ,]
test_data = merged_data[-index ,]

#Removing the columns that will not be part of further analysis
train_data = train_data[ , c(1,2,3,4,5) := NULL]
test_data = test_data[ , c(1,2,3,4,5) := NULL]

#Converting to data frame for better compatibility
data_f = as.data.frame(train_data)
data_f_test = as.data.frame(test_data)

#Null Value treatment
data_f = as.data.frame(apply(data_f, MARGIN = c(1,2), function(x) ifelse(is.na(x),0,x)))

data_f_test = as.data.frame(apply(data_f_test, MARGIN = c(1,2), function(x) ifelse(is.na(x),0,x)))


######################## Model: XgBoost (Cross Validation) ##############################################


sparse_matrix <- sparse.model.matrix(is_duplicate ~ .-1, data = data_f)
y  = data_f$is_duplicate


searchGridSubCol =expand.grid(
  nrounds = 10,
  eta = c(0.01, 0.001, 0.0001),
  gamma = c(0 , 1, 1.5),
  subsample = c(0.5, 0.75, 1),
  colsample_bytree = c(0.3, 0.4 , 0.5, 0.8, 1),
  nfolds = 5
)



#Xg- boost on cross validation 

xgboost_validation <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  SubsampleRate <- parameterList[["subsample"]]
  ColsampleRate <- parameterList[["colsample_bytree"]]
  ntrees <- parameterList[["nfolds"]]
  nfolds_current  = parameterList[["nrounds"]]
  eta_curr = parameterList[["eta"]]
  gamma_curr =  parameterList[["gamma"]]
  
  
  
  xgboostModelCV <- xgb.cv(data =  sparse_matrix, label = y,
                           nrounds = ntrees, nfold = nfolds_current, showsd = TRUE, 
                           metrics = "rmse", verbose = TRUE, "eval_metric" = "rmse",
                           "objective" = "binary:logistic", max_depth = 13,
                           "subsample" = SubsampleRate, "colsample_bytree" = ColsampleRate, 
                           eta = eta_curr , gamma =  gamma_curr)
  
  xvalidationScores <- as.data.frame(xgboostModelCV)
  #Save rmse of the last iteration
  rmse <- tail(xvalidationScores$test.rmse.mean, 1)
  
  return(c(rmse, SubsampleRate, ColsampleRate))
  
})





######################## Model: Random Forest (Cross Validation) ##############################


control <- trainControl(method="repeatedcv", number=2, repeats=1)
metric <- "rmse"
set.seed(203)
mtry <- sqrt(ncol(data_f))
tunegrid <- expand.grid(.mtry=mtry)
randm <- train(is_duplicate ~ ., data=data_f, method="cforest", metric=metric, 
               tuneGrid=tunegrid, trControl=control)









