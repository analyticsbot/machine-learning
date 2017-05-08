#Joining all the features together
## USing data table for more efficient and quicker processing

library(data.table)

#Loading all the features data 
dt <- fread("sentiment.csv"  , stringsAsFactors =  F , header = T)
dt1 <- fread("Interrogative_feat.csv"  , stringsAsFactors =  F , header = T) 
dt2 <- fread("train_features.csv"  , stringsAsFactors =  F , header = T) 
dt3 <- fread("train_features2.csv"  , stringsAsFactors =  F , header = T) 
dt4 <- fread("train_features3.csv"  , stringsAsFactors =  F , header = T) 
dt5 <- fread("train_features4.csv"  , stringsAsFactors =  F , header = T)
dt6 <- fread("train_features5.csv"  , stringsAsFactors =  F , header = T)
dt7 <- fread("train_features6.csv"  , stringsAsFactors =  F , header = T)


#Remvoing the serial number column 

dt = dt[,  (1) := NULL]
dt1 = dt1[,  (1) := NULL]
dt2 = dt2[,  (1) := NULL]
dt3 = dt3[,  (1) := NULL]
dt4 = dt4[,  (1) := NULL]
dt5 = dt5[,  (1) := NULL]

# Data check = Checking unique set of values for each of the parts of data 
merged_data  = merge(dt , dt1[ , (c('qid1','qid2','question1','question2','is_duplicate')) := NULL ] 
                     , by = ("id") , all = F)
merged_data  = merge(merged_data , dt2[ , (c('qid1','qid2','question1','question2','is_duplicate')) := NULL ] 
                     , by = ("id") , all = F)
merged_data  = merge(merged_data , dt3[ , (c('qid1','qid2','question1','question2','is_duplicate')) := NULL ] 
                     , by = ("id") , all = F)
merged_data  = merge(merged_data , dt4[ , (c('qid1','qid2','question1','question2','is_duplicate')) := NULL ] 
                     , by = ("id") , all = F)
merged_data  = merge(merged_data , dt5[ , (c('qid1','qid2','question1','question2','is_duplicate')) := NULL ] 
                     , by = ("id") , all = F)
merged_data  = merge(merged_data , dt6[ , (c('qid1','qid2','question1','question2','is_duplicate')) := NULL ] 
                     , by = ("id") , all = F)
merged_data  = merge(merged_data , dt7[ , (c('qid1','qid2','question1','question2','is_duplicate')) := NULL ] 
                     , by = ("id") , all = F)


# Clearing memory by removing redundant data sets
remove( list = c('dt', 'dt1', 'dt2' , 'dt3' , 'dt4' , 'dt5' , 'dt6' , 'dt7'))








