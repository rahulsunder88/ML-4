# ML4 - HackerEarth
library(caret)
library(randomForest)
library(Matrix)
library(xgboost)
library(e1071)
library(caTools)
setwd("~/Downloads/c771afa0-c-HackerEarthML4Updated")
set.seed(100)
library(data.table)
train <- fread("train_data.csv",stringsAsFactors = FALSE,strip.white=TRUE,sep=",")
test <- fread("test_data.csv",stringsAsFactors = FALSE,strip.white=TRUE,sep=",")

train$type = 'train'
test$type = 'test'
test$target <- as.integer(0)

train <- rbind(train,test)

#----------------------------------------- Feature Engineering ------------------------#
train$new02 = ' '
train$new02 = ifelse(train$cont_16 >= 0.4,10,11)

table(train[train$cont_16 >= 0.4 & cont_17 >= 0.75,]$target)

train$cont_4_8 = (train$cont_4 * train$cont_8)
train$cont_6_8 = (train$cont_6 * train$cont_8)
train$cont_4_6_8 = (train$cont_4 * train$cont_6 *  train$cont_8)
#train$cat_20_21 = (train$cat_20 * train$cat_21 )
#train$cat_2_20_21 = (train$cat_2 * train$cat_20 * train$cat_21 )
train$cat_mean_2t = ((train$cat_20 + train$cat_21 + train$cat_22 + train$cat_23)/(mean(train$cat_20) + 
                                                                                    mean(train$cat_21) + mean(train$cat_22) + mean(train$cat_23)))
train$cat_mean_123 = ((train$cont_2 + train$cont_3 )/(mean(train$cont_1) + 
                                                        mean(train$cont_2) + mean(train$cont_3) ))

train$cat_mean_23 = ((train$cont_2 + train$cont_3 )/mean(train$cont_3) )

train$cont_8new = train$cont_8
train[train$cont_8 <= 0.014,]$cont_8new = 2
train[train$cont_8 > 0.014 & train$cont_8 < 1,]$cont_8new = 1
train[train$cont_8 == 1,]$cont_8new = 0

train$cont_9new = train$cont_9
train[train$cont_9 == 0,]$cont_9new = 0
train[train$cont_9 <= 0.08 & train$cont_9 > 0,]$cont_9new = 1
train[train$cont_9 > 0.08,]$cont_9new = 1

train$cont_10new = train$cont_10
train[train$cont_10 == 0,]$cont_10new = 0
train[train$cont_10 > 0 & train$cont_10 <=1,]$cont_10new = 1

train$cont_11new = train$cont_11
train[train$cont_11 == 1,]$cont_11new = 0
train[train$cont_11 >= 0 & train$cont_11 <1,]$cont_11new = 1

train$cont_12new = train$cont_12
train[train$cont_12 == 0,]$cont_12new = 0
train[train$cont_12 > 0 & train$cont_12 < 0.1,]$cont_12new = 2
train[train$cont_12 >= 0.1 & train$cont_12 <= 1,]$cont_12new = 1

train$cont_13new = train$cont_13
train[train$cont_13 == 0 ,]$cont_13new = 2
train[train$cont_13 > 0 & train$cont_13 <=0.05,]$cont_13new = 1
train[train$cont_13 > 0.05 & train$cont_13 <=1,]$cont_13new = 0

train$cont_14new = train$cont_14
train[train$cont_14 == 0,]$cont_14new = 0
train[train$cont_14 > 0,]$cont_14new = 1

train$cont_15new = train$cont_15
train[train$cont_15 == 0,]$cont_15new = 0
train[train$cont_15 == 1,]$cont_15new = 2
train[train$cont_15 > 0 & train$cont_15 <1,]$cont_15new = 1

train$cont_16new = train$cont_16
train[train$cont_16 == 0,]$cont_16new = 0
train[train$cont_16 == 1,]$cont_16new = 2
train[train$cont_16 > 0 & train$cont_16 <1,]$cont_16new = 1

train$cat_2new = train$cat_2
train[train$cat_2 != 10,]$cat_2new = 0
train[train$cat_2 == 10,]$cat_2new = 1

#----------------------------------------- End of Feature Engineering ------------------------#

test = train[train$type == 'test',]
train = train[train$type == 'train',]

test$target = NULL
test$type = NULL
train$type = NULL
#------------------------------------------ Remove outliers ------------------------------------#
train = train[train$cat_10 < 100,] 
train = train[train$cat_14 < 10,] 
train = train[train$cat_18 < 1,] 
train = train[train$cont_1 < 20000,] 
train = train[train$cont_2 < 100000,] 
train = train[train$cont_3 < 100000,] 
train = train[train$cont_14 < 0.6,] 
train = train[train$cat_2 < 62,]
train = train[train$cat_4 < 0.1,]
train = train[train$cat_5 < 1.5,]
train = train[train$cat_6 < 0.5,]
# train = train[train$cat_7 < 25,]
train = train[train$cat_8 < 0.2,]
# train = train[train$cat_4 < 1,]
# train = train[train$cat_5 < 1,]
train = train[train$cat_10 <= 1,]
train = train[train$cat_12 < 0.2,]
train = train[train$cat_13 < 6,]
train = train[train$cat_14 < 2,]
train = train[train$cat_15 <= 1,]
train = train[train$cat_16  < 1 ,]
# train = train[train$cont_4_6_8  <= 0.02 ,]
# train = train[train$cat_2_20_21 < 3000000 ,]

#------------------------------------------ End of Remove outliers ------------------------------------#

# kmeans = kmeans(x = train[,-c('new02','cont_4_8','cont_4_6_8','cat_20_21',
#                              'cat_2_20_21','cat_mean_2t','cat_mean_123')], centers = 2)
kmeans = kmeans(x = train[,2:42], centers = 3)
y_kmeans = kmeans$cluster
train$y_kmeans = y_kmeans
set.seed(29)
test_kmeans = kmeans(x = test[,2:42], centers = 3)
y_test_kmeans = test_kmeans$cluster
test$y_kmeans = y_test_kmeans

# plot details
# cont_1 - 1
# cont_2 - none
# cont_3 - 1
# cont_4 - 0 & 2
# cont_5 - 1
# cont_6 - 0 & 2
# cont_7 - 0 & 1
# cont_8 - 0 & 2
# cont_9 - 0 , 1 & 2
# cont_10 - 1
# cont_11 - 1 very strong then 0 then 2
# cont_12 - 0,1 & 2
# cont_13 - 1 very strong then 0 then 2
# cont_14 - until 0.6 1 very strong others are sparse
# cont_15 - 0, 1 & 2
# cont_16 - 2 after 0.4, 0 is uniform, 1 < 0.5
# cont_17 - 1 very strong, 0 very strong, 2 very strong after 0.75
# cont_18 - 1 very strong

# cont_16
# cat
#  cat_1  - no sig
#  cat_2  - 0 & 2
#  cat_3  - 0, 2 then 1
#  cat_4  - 0 & 2 at value 1
#  cat_5  - no sig
#  cat_6  - 1 at point 2
#  cat_7  - 1
#  cat_8  - no sig
#  cat_9  - no sig
#  cat_10  - outliers for 1 after 200
#  cat_11  - no sig
#  cat_12  - 1 at and after 1
#  cat_13  - outliers for 1 after 200
#  cat_14  - outliers for 1 after 10
#  cat_15  - 2 and 1 at point 1
#  cat_16  - 1
#  cat_17  - no sig
#  cat_18  - outlier for 2 at 1
#  cat_19  - no sig
#  cat_20  - 0 and 2
#  cat_21  - 0 and 2
#  cat_22  - 1
#  cat_23  - 1, 0 then 2



# Create the function.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Tuning the parameters #
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3)

xgb.grid <- expand.grid(nrounds = 500,
                        max_depth = c(100,150,200,250,500),
                        eta = c(0.01,0.3, 1),
                        gamma = c(0.0, 0.2, 1),
                        colsample_bytree = c(0.5,0.8, 1),
                        min_child_weight=seq(1,10)
)

getXGBData <- function(train, test)
{
  split <- sample.split(Y = train$target, SplitRatio = 0.5)
  
  dtrain <- train[split]
  dvalid <- train[!split]
  
  dtrain <- xgb.DMatrix(data = as.matrix(dtrain[,-c('connection_id','target'), with=F]), label = dtrain$target)
  dvalid <- xgb.DMatrix(data = as.matrix(dvalid[, -c('connection_id', 'target'), with = F]), label = dvalid$target)
  dtest <- xgb.DMatrix(data = as.matrix(test[,-c('connection_id'), with=F]))
  
  return (list(train = dtrain, test = dtest, eval = dvalid))
  
}

getMulAcc <- function(pred, dtrain)
{
  label <- getinfo(dtrain, "label")
  acc <- mean(label == pred)
  return(list(metric = "maccuracy", value = acc))
}


runModels <- function(dtrain, dtest, dvalid, XGB = 0) ## set 1 for XGB: Default run is RF
{
  
  if(XGB == 1)
  {
    cat('Running Gradient Boosting Model...\n\n')
    # default parameters
    params <- list(objective = 'multi:softmax',
                   num_class = 3)
    
    watchlist <- list('train' = dtrain, 'valid' = dvalid)
    clf <- xgb.train(params
                     ,dtrain
                     ,1000
                     ,watchlist
                     ,feval = getMulAcc
                    # ,eval_metric = "auc"
                     ,print_every_n = 20
                     ,early_stopping_rounds = 30
                     ,maximize = T
                     ,trControl=cv.ctrl
                     ,tuneGrid=xgb.grid
    )
    
    
    pred <- predict(clf, dtest)
    
  } else if (XGB == 0) {
    
    cat('Running Random Forest Model...\n\n')
    params <- list(booster = 'dart'
                   ,objective = 'multi:softmax'
                   ,num_class = 3
                   ,normalize_type = 'tree'
                   ,rate_drop = 0.1)
    
    watchlist <- list('train' = dtrain, 'valid' = dvalid)
    clf <- xgb.train(params
                     ,dtrain
                     ,1000
                     ,watchlist
                     ,feval = getMulAcc
                     ,print_every_n = 20
                     ,early_stopping_rounds = 30
                     ,maximize = T
                     ,trControl=cv.ctrl
                     ,tuneGrid=xgb.grid
    )
    
    
    pred <- predict(clf, dtest)
    
  } 
  
  return(pred)
  
}


train_01 = rbind(train[train$target==0,],train[train$target==1,])
train_12 = rbind(train[train$target==1,],train[train$target==2,])
train_02 = rbind(train[train$target==0,],train[train$target==2,])

xgbdata <- getXGBData(train, test)

predsRF <- runModels(dtrain = xgbdata$train, dtest = xgbdata$test, dvalid = xgbdata$eval)
predsXGB <- runModels(dtrain = xgbdata$train, dtest = xgbdata$test, dvalid = xgbdata$eval,XGB = 1)

xgbdata_01 <- getXGBData(train_01, test)
predsRF_01 <- runModels(dtrain = xgbdata_01$train, dtest = xgbdata_01$test, dvalid = xgbdata_01$eval)
predsXGB_01 <- runModels(dtrain = xgbdata_01$train, dtest = xgbdata_01$test, dvalid = xgbdata_01$eval,XGB = 1)

xgbdata_12 <- getXGBData(train_12, test)
predsRF_12 <- runModels(dtrain = xgbdata_12$train, dtest = xgbdata_12$test, dvalid = xgbdata_12$eval)
predsXGB_12 <- runModels(dtrain = xgbdata_12$train, dtest = xgbdata_12$test, dvalid = xgbdata_12$eval,XGB = 1)

xgbdata_02 <- getXGBData(train_02, test)
predsRF_02 <- runModels(dtrain = xgbdata_02$train, dtest = xgbdata_02$test, dvalid = xgbdata_02$eval)
predsXGB_02 <- runModels(dtrain = xgbdata_02$train, dtest = xgbdata_02$test, dvalid = xgbdata_02$eval,XGB = 1)

#=============== predict using Random forest ==============================#
set.seed(1234)
samplesize = (0.50 * nrow(train_01))
sampledata = sample(seq_len(nrow(train_01)),size=samplesize)
train_train <- train_01[sampledata,]
train_test <- train_01[-sampledata,]

classifier = randomForest(as.factor(target) ~ ., train_train[,-c('connection_id')],mtry=26,
                          ntree = 100)

y_pred = predict(classifier, newdata = train_test[,-c('connection_id')])
confusionMatrix(y_pred,train_test$target)

set.seed(1234)
samplesize = (0.50 * nrow(train_12))
sampledata = sample(seq_len(nrow(train_12)),size=samplesize)
train_train <- train_12[sampledata,]
train_test <- train_12[-sampledata,]
classifier_12 = randomForest(as.factor(target) ~ ., train_train[,-c('connection_id')],mtry=26,
                             ntree = 100)

# Predicting the Test set results
y_pred_12 = predict(classifier_12, newdata = train_test[,-c('connection_id')])

# Making the Confusion Matrix
cm = table(y_pred_12,train_test$target)
cm
confusionMatrix(y_pred_12,train_test$target)

set.seed(1234)
samplesize = (0.50 * nrow(train_02))
sampledata = sample(seq_len(nrow(train_02)),size=samplesize)
train_train <- train_02[sampledata,]
train_test <- train_02[-sampledata,]
classifier_02 = randomForest(as.factor(target) ~ ., train_train[,-c('connection_id')],mtry=26,
                             ntree = 100)

# Predicting the Test set results
y_pred_02 = predict(classifier_02, newdata = train_test[,-c('connection_id')])

# Making the Confusion Matrix
cm = table(y_pred_02,train_test$target)
cm
confusionMatrix(y_pred_02,train_test$target)

y_pred <- predict(classifier,newdata=test)
y_pred_12 <- predict(classifier_12,newdata=test)
y_pred_02 <- predict(classifier_02,newdata=test)

#=============== predict using Logistic regression ===============#
set.seed(1234)
samplesize = (0.50 * nrow(train_01))
sampledata = sample(seq_len(nrow(train_01)),size=samplesize)
train_train <- train_01[sampledata,]
train_test <- train_01[-sampledata,]
train_train = train_train[,-c('connection_id')]
train_test = train_test[,-c('connection_id')]

myLogistic_01 <- glm(target ~.
                     , data = train_train
                     ,family = binomial(link = "logit"))

logisticprediction_01 <- predict(myLogistic_01,newdata=train_test,type = 'response')
logisticprediction_01 = round(logisticprediction_01)
confusionMatrix(logisticprediction_01,train_test$target)

set.seed(1234)
samplesize = (0.50 * nrow(train_12))
sampledata = sample(seq_len(nrow(train_12)),size=samplesize)
train_train <- train_12[sampledata,]
train_test <- train_12[-sampledata,]
train_train = train_train[,-c('connection_id')]
train_test = train_test[,-c('connection_id')]

myLogistic_12 <- glm(as.factor(target) ~.
                     , data = train_train
                     ,family = binomial(link = "logit"))

logisticprediction_12 <- predict(myLogistic_12,newdata=train_test,type='response')
logisticprediction_12 = round(logisticprediction_12)
logisticprediction_12 = ifelse(logisticprediction_12 == 0,1,2)
confusionMatrix(logisticprediction_12,train_test$target)

set.seed(1234)
samplesize = (0.50 * nrow(train_02))
sampledata = sample(seq_len(nrow(train_02)),size=samplesize)
train_train <- train_02[sampledata,]
train_test <- train_02[-sampledata,]
train_train = train_train[,-c('connection_id')]
train_test = train_test[,-c('connection_id')]

myLogistic_02 <- glm(as.factor(target) ~.
                     , data = train_train
                     ,family = binomial(link = "logit"))

logisticprediction_02 <- predict(myLogistic_02,newdata=train_test,type='response')
logisticprediction_02 = round(logisticprediction_02)
logisticprediction_02 = ifelse(logisticprediction_02 == 0,0,2)
confusionMatrix(logisticprediction_02,train_test$target)

logisticprediction_01 <- predict(myLogistic_01,newdata=test,type = 'response')
logisticprediction_01 = round(logisticprediction_01)
logisticprediction_12 <- predict(myLogistic_12,newdata=test,type='response')
logisticprediction_12 = round(logisticprediction_12)
logisticprediction_12 = ifelse(logisticprediction_12 == 0,1,2)
logisticprediction_02 <- predict(myLogistic_02,newdata=test,type='response')
logisticprediction_02 = round(logisticprediction_02)
logisticprediction_02 = ifelse(logisticprediction_02 == 0,0,2)
#===============  END Of Logistic regression ===============#

set.seed(1234)
samplesize = (0.50 * nrow(train))
sampledata = sample(seq_len(nrow(train)),size=samplesize)
train_train <- train[sampledata,]
train_test <- train[-sampledata,]
train_train = train_train[,-c('connection_id')]
train_test = train_test[,-c('connection_id')]

formula1 = (as.factor(target) ~ . )

myNaives <- naiveBayes(formula =formula1
                       , data = train_train)


bayesprediction <- predict(myNaives,newdata =train_test,type = 'class')
table(bayesprediction,train_test$target)
confusionMatrix(bayesprediction,train_test$target)

naive_prediction <- predict(myNaives,newdata=test,type = 'class')


#====================== model ensembling 1
ensemble_dataframe = data.frame(predsRF_01,predsXGB_01,predsRF_12,predsXGB_12,predsRF_02,predsXGB_02,
                                logisticprediction_02,logisticprediction_12,logisticprediction_01
                                ,y_pred_12,y_pred_02,y_pred)

b = apply(ensemble_dataframe,1,getmode)
table(b)

#----------------------- best method
ensemble_dataframe31 = data.frame(predsXGB,naive_prediction,ensemble_dataframe)

b31 = apply(ensemble_dataframe31,1,getmode)
table(b31)
sampsub31 <- fread("sample_submission.csv")
sampsub31[, target := b31]
fwrite(sampsub3, "sub31.csv")

#----------------------- End of best method
