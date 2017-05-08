#Logistic Regression Model on the extracted features

install.packages("glmulti")
library(glmulti)
colnames(merged_data)
cor(merged_data[3:102])

#merged_data_new = merged_data[,c(2,29,30,33,34,37,38,40,71:102)]
merged_data_new = merged_data[,c(2,71,72,73,86:102)]

#Splitting the dataset into train and test
test_train_split = function(frac, merged_data){
  
  classes = merged_data_new$is_duplicate
  set.seed(203)
  trainingRows <- createDataPartition(classes,p = frac,list= FALSE)
  return(trainingRows)
}

index = test_train_split(10000/400000, merged_data_new)

#Creating test and training set
train_data = merged_data_new[index ,]
test_data = merged_data_new[-index ,]


data_f = as.data.frame(apply(train_data, MARGIN = c(1,2), function(x) ifelse(is.na(x),0,x)))


#Selecting the Best Model using glmulti package
  lmtest =
    glmulti(is_duplicate ~ ., data = data_f,
            level = 1,               # No interaction considered
            method = "h",            # Exhaustive approach
            crit = "aic",            # AIC as criteria
            confsetsize = 5,         # Keep 5 best models
            plotty = F, report = F,  # No plot or interim reports
            fitfunction = "lm")      # lm function

## Show 5 best models (Use @ instead of $ for an S4 object)
lmtest@formulas

colnames(data_f)

#Logistic model using glm function
lmfinal = glm(is_duplicate~.,data = data_f,family = binomial(link='logit'))
summary(lmfinal)
Predicted_values = ifelse(lmfinal$fitted.values > 0.6,1,0)
table(Predicted_values,data_f$is_duplicate)

#Plotting the ROC curve
ROC <- roc(response = data_f$is_duplicate,
               predictor = lmfinal$fitted.values,
               levels = levels(factor(data_f$is_duplicate)))
plot(ROC, type="S")

#Logistic model 2 using significant features
lmfinal1 = glm(is_duplicate~ cosineSimilarity  + jaccardSimilarity + glove_sim  + word2vec_sim + Wratio + Part_ratio
              ,data = data_f,family = binomial(link='logit'))

Predicted_values = ifelse(lmfinal1$fitted.values > 0.6,1,0)
table(Predicted_values,data_f$is_duplicate)

#Plotting the ROC curve
ROC <- roc(response = data_f$is_duplicate,
           predictor = lmfinal1$fitted.values,
           levels = levels(factor(data_f$is_duplicate)))
plot(ROC, type="S")

#Prediciting the values in test dataset and calculating the accuracy
letstry = predict(lmfinal1 , newdata = test_data)
Predicted_values_letstry = ifelse(letstry > 0.6,1,0)
prop.table(table(Predicted_values_letstry , test_data$is_duplicate ))
