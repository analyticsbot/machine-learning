###################################################################################
# The problem statement - Quora Duplicate Question Prediction
# This R code extracts feature from the question pairs which will then be fed
# to a machine learning model for predictions
###################################################################################

## to begin, it is required to install these packages
install.packages(c("dplyr", "xgboost"))

## load the required packages
library(xgboost)
library(readr)
library(dplyr)

############# Distributing data for train and test#############
### this function is used to split the dataset into training and testing dataset
test_train_split = function(frac, merged_data){
  
  ## this function will split the dataset in the same proportion as in the
  ## original class. so as the modelling is done correctly
  classes = merged_data$is_duplicate
  set.seed(203)
  trainingRows <- createDataPartition(classes,p = frac,list= FALSE)
  return(trainingRows)
}

## read the merged dataset with all the features
merged_data = read.csv('~/jeremy/quora/merged_data.csv', stringsAsFactors = F, header = T)

## check the class of the merged file
class(merged_data)

## check the names of the columns in the file
names(merged_data)

## check the structure of the data
str(merged_data)


## THIS CODE DEALS WITH FIRST TUNING THE XGBOOST MODEL
## AND THEN TRAINING THE MODEL AND PREDICTING THE OUTPUT


### STEP 1 -- TUNING THE MODEL
## get the index of indexes for train and test dataset
index = test_train_split(0.7, merged_data)

#Removing the columns that will not be part of further analysis
merged_data1 = select(merged_data, -c(1:2))

## TRAINING DATASET IS 70% OF TRAIN & 30% OF TEST DATA
#Creating test and training set
train_data = merged_data[index ,]
test_data = merged_data[-index ,]

## Null Value treatment
## CHANGE THE NULL VALUES TO 0
data_f = as.data.frame(apply(data_f, MARGIN = c(1,2), function(x) ifelse(is.na(x),0,x)))
data_f_test = as.data.frame(apply(data_f_test, MARGIN = c(1,2), function(x) ifelse(is.na(x),0,x)))

######################## Model: XgBoost (Cross Validation) ##############################################

## CHECK THE DIMENSION TO MAKE SURE NO ROWS ARE LOST
dim(data_f)

## XGBOOST ONLY DEALS WITH SPARSE MATRIX AND HENCE THE DATASET NEEDS TO BE CONVERTED
## TO SPARSE MATRIX
sparse_matrix <- sparse.model.matrix(is_duplicate ~ .-1, data = data_f)
sparse_matrix_2 <- sparse.model.matrix(is_duplicate ~ .-1, data = test_data_1[,-1])

## TAKE OUT THE TARGET VARIABLE FROM THE DATASET
y  = data_f$is_duplicate

## TAKE OUT THE DATASET FROM THE TEST DATASET WHICH WOULD BE USED TO READ THE ACCURACY
y_true = test_data_1$is_duplicate

## INITIALIZE A SEARCH GRID WITH THE PARAMETERS THAT WE HAVE TO TUNE
## NROUNDS IS THE NUMBER OF TREES WE HAVE TO BUILT. THE CODE WILL STOP BEFORE IF THERE
## IS NOT IMPROVEMENT IN THE ERROR
## ETA IS THE LEARNING RATE. LOWER LEARNING RATE IS GOOD BUT IS SLOW. HIGHER IS FAST BUT NOT GOOD
## GAMMA IS ANOTHER TUNING PARAMETER
## SUBSAMPLE IS HOW MANY SAMPLE OF ROWS SHOULD BE TAKEN FOR EACH OF THE TREES. THIS WOULD
## SELECT THAT MUCH PERCENTAGE OF ROWS FOR THE MODEL BUILDING
## COLSAMPLE_BYTREE IS HOW MANY COLUMNS SHOULD BE RANDOMLY SELECTED
## MAX_DEPTH IS THE MAXIMUM DEPTH OF THE TREES SHOULD BE BUILT
## EARLY.STOP.ROUND WILL stop if the performance keeps getting worse consecutively for k rounds.
searchGridSubCol =expand.grid(
  nrounds = 5000,
  eta = c(0.01, 0.001),
  gamma = c(1, 1.5),
  subsample = c(0.5, 0.8),
  colsample_bytree = c(0.5, 1),
  max_depth = c(5, 8),
  nfolds = 5,
  early.stop.round = 50
)

## RUNNING AN APPLY FUNCTION ON ALL THE PARAMETERS COMBINATIONS THAT WAS TO BE COMPARED
#Xg- boost on cross validation 

## THIS FUNCTION WOULD RUN ALL THE PARAMETER COMBINATIONS AND THEN RETURN A DATAFRAME
## WITH THE ERROR VALUES AND THE PARAMETER COMBINATION WHICH WOULD THEN BE SORTED
## TO GET THE BEST PERFORMING PARAMETERS
xgboost_validation <- apply(searchGridSubCol, 1, function(parameterList){
  
  #Extract Parameters to test
  SubsampleRate <- parameterList[["subsample"]]
  ColsampleRate <- parameterList[["colsample_bytree"]]
  ntrees <- parameterList[["nrounds"]]
  nfolds_current  = parameterList[["nfolds"]]
  eta_curr = parameterList[["eta"]]
  gamma_curr =  parameterList[["gamma"]]
  mx_depth = parameterList[['max_depth']]
  early.stop.round = parameterList[['early.stop.round']]

  ## THIS CROSS VALIDATION WOULD RUN THE TREES AND STOP ONCE THE ERROR 
  ## STOPS DECREASING FOR SOME ROUNDS
  ## AT THE END WE TAKE THE PARAMETER LIST AND THE ERROR VALUE
  ## WHICH WOULD LATER BE COMPARED
  xgboostModelCV <- xgb.cv(data =  sparse_matrix, label = y,
                           nrounds = ntrees, nfold = nfolds_current, showsd = TRUE, 
                           metrics = "error", verbose = T, eval_metric = "error",
                           objective = "binary:logistic", max_depth = mx_depth,
                           subsample = SubsampleRate, colsample_bytree = ColsampleRate, 
                           eta = eta_curr , gamma =  gamma_curr, early.stop.round = early.stop.round )
  
  ## TAKE OUT THE PARAMETERS USED IN THE MODEL
  xvalidationScores <- as.data.frame(xgboostModelCV$params)
  
  ## TAKE OUT THE ERROR ON TRAIN AND VALIDATION SET
  error = as.data.frame(xgboostModelCV$evaluation_log)[dim(as.data.frame(xgboostModelCV$evaluation_log))[1], ]
  
  ## SAVE THE ERROR AND PARAMETERS
  output <- cbind(xvalidationScores, error)
  
  ## RETURN THE SPECIFIC VALUES
  return(data.frame(output, SubsampleRate=SubsampleRate, ColsampleRate=ColsampleRate,
                    eta_curr=eta_curr, gamma_curr=gamma_curr, mx_depth=mx_depth))
  
})

## TAKE THE RESULT AND CONVERT TO DATAFRAME
df <- data.frame(matrix(unlist(xgboost_validation), nrow=32, byrow=T))
df <- df[order(df[,11])]

## AFTER CROSS VALIDATION, THE BEST RESULTS WERE FOUND FOR THE FOLLOWING VALUES
## SUBSAMPLERATE = 0.8 
## COLSAMPLERATE = 0.5 
## ETA (LEARNING RATE) = 0.010 
## GAMMA = 1.0
## MAX DEPTH = 8
## TRAINING ROUNDS = 4934

## THE VALUES WILL BE USED TO TRAIN THE PARAMETERS
xgboostModel <- xgboost(data = sparse_matrix_1, 
                        label = y, 
                        eta = 0.01,
                        max_depth = 8, 
                        nround=4934, 
                        subsample = 0.8,
                        colsample_bytree = 0.5,
                        seed = 1,
                        eval_metric = "error",
                        objective = "binary:logistic")

## USE THE MODEL TO PREDICT THE LABELS FOR TEST DATASET
pred_prob = predict(xgboostModel, newdata = sparse_matrix_2)
pred = pred_prob>0.6
prop.table(table(pred, y_true))
# q               0         1
# FALSE 0.6195667 0.3765333
# TRUE  0.0019000 0.0020000

## FINAL STEP IS TO SEE WHICH OF THE FEATURES THAT WERE EXTRACTED ARE USEFUL
## AND THEIR RELATIVE IMPORTANCE
importance <- xgb.importance(feature_names = sparse_matrix_1@Dimnames[[2]], model = xgboostModel)
head(importance, 10)
# > head(importance, 10)
# Feature       Gain      Cover  Frequency
# 1: numCommonWordsNormalized 0.15083767 0.05193972 0.03209841
# 2:             word2vec_sim 0.07641091 0.06571602 0.04022066
# 3:                glove_sim 0.06618036 0.06096863 0.03930985
# 4:           numCommonWords 0.03422043 0.01777784 0.01345115
# 5:         cosineSimilarity 0.02898954 0.04481639 0.03628842
# 6:      manhattanSimilarity 0.02232537 0.01148825 0.01189825
# 7:        jaccardSimilarity 0.02224680 0.03061215 0.02714289
# 8:                        X 0.02121544 0.03362651 0.03173909
# 9:      minkowskiSimilarity 0.01971192 0.01491479 0.01941277
# 10:        numCharByWordDiff 0.01756334 0.02574115 0.02287008
# > 
