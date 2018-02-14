train = read.csv("train.csv")      # load data
train$ID = NULL
#Removing ID as it plays no role in the machine learning process
dim(train)

#Removing erratic values 
train[train==-999999] <- train[train==9999999999] <- NA
sum(is.na(train))
train <- na.omit(train)
sum(is.na(train))

train.y = as.factor(train$TARGET); train$TARGET = NULL  # Extracting the response - TARGET#

###Cleaning the dataset - Part_1
# Removing 70 predictors having constant 0 value across all observations
for (f in setdiff(names(train),c('TARGET'))) {
  if (mean(train[[f]])== sum(train[[f]])) {
     #cat(f, "is constant in train.\n")
    train[[f]] = NULL
      }
}
rm(f)

library(digest)
# Removing 26 duplicated predictors.
dup = duplicated(lapply(train, digest))
train = train[!dup]
rm(dup)

# 0 count per row
count0 <- function(x) return( sum(x == 0) )
train$n0 = apply(train, 1, FUN=count0)   
#making a new feature, thats the count of the number of zeros in the row
#boxplot(train$n0~train.y ,main= "Count of Zeros vs Customer Satisfaction")

library(caret)
# Removing features being a specific linear function of other features
lin.comb = findLinearCombos(train)
train = train[, -lin.comb$remove];  rm(lin.comb)

# Removing the correlated features
cor.features = findCorrelation(cor(train), cutoff = .95, verbose = FALSE)
train = train[, -cor.features];        rm(cor.features)

#Removing duplicate rows
dup=!duplicated(train);
train.y = train.y[dup]
train= train[dup,] ; rm(dup)# removes duplicated rows
dim(train)
table(train.y)

# Ranking features by importance
if(F)  #put T , to run the following chuck
{
#For each feature, the % of rows that have the most frequent value.
feature <- most_freq_percent <- numeric(length(names(train)))
for (i in 1:length(names(train))) {
  tabl = as.data.frame(table(train[[names(train)[i]]]))
  tabl = tabl[order(tabl$Freq,decreasing=TRUE),]
  #cat(names(train)[i] ,"\n")
  feature[i] = names(train)[i]
  most_freq_percent[i] = round(100.0 * tabl[1,2] / nrow(train),5)
}
percent= data.frame(feature , most_freq_percent ); rm(feature , most_freq_percent)
write.csv(file="unique_importance.csv", x=percent , row.names = T)

train$TARGET = train.y; rm(train.y)

# Chi Test to find the ranking of the variables.  
chi_weights <- chi.squared(TARGET~., train)
write.csv(file="Chi_squared_importance.csv", x=importance , row.names = T)

# Relief algorithim to find the ranking of the variables.  
system.time(relief_weights <- relief(TARGET~., data=train, neighbours.count = 5, sample.size = 5))
write.csv(file="Relief_weights_importance.csv", x=importance , row.names = T)

# Random Forest to find the ranking of the variables.  
system.time(forest <- randomForest(TARGET ~ .,data=train , importance =TRUE))
importance <- forest$importance
write.csv(file="Random_forest_importance.csv", x=importance , row.names = T)
varImpPlot(forest)}

#importing feature names with importance
fsel=read.csv("Name_importance.csv" , row.names = 1)
name=row.names(fsel[fsel$importance>1,]);  rm(fsel)

train = train[,c(name)] ;rm(name)  # feature selection based on the ranking

# removes duplicated rows
dup=!duplicated(train)
train.y = train.y[dup]
train= train[dup,] ; rm(dup)
dim(train)
table(train.y)

train$TARGET =train.y; rm(train.y)
#Jumbling the order of the rows in the Dataset and this data set will be used for CV purposes
set.seed(1);  
train = train[sample(nrow(train)),]
train_0 = train[train$TARGET== 0,]
train_0$folds <- cut(1:nrow(train_0),breaks=10,labels=FALSE)  # the 10 folds to be used for the 10-fold cross validation throughout this report.
train_1 = train[train$TARGET==1,]
train_1$folds <- cut(1:nrow(train_1),breaks=10,labels=FALSE)

train = rbind(train_0,train_1); rm(train_0 , train_1)
folds = train$folds; train$folds =NULL
table(folds)

library(ROCR)
#Function that returns ROC plot and the AUC
ROCplot =function(probabilty ,test_y , ...){
predob = prediction(probabilty , test_y)
perf = performance (predob , measure = "tpr", x.measure = "fpr")
auc= performance(predob, measure = "auc")
plot(perf,...)
abline(a=0, b= 1, lty=3)
return(round(auc@y.values[[1]],3))
}
####################
#### MODELING ######
####################

#Logistic Regression
#without sampling
CV_AUC_LG = numeric(10)
for(i in 1:10){
  cat(i)
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    pred.model = glm( TARGET ~  . , data = train[-testIndexes,], family = "binomial")
    buffer = predict(pred.model , train[testIndexes,] , type = "response")
    CV_AUC_LG[i] = ROCplot(buffer , train$TARGET[testIndexes],main ="Logistic Regression")
}
rm(buffer,pred.model,testIndexes,i)
mean(CV_AUC_LG)

# with oversampling
CV_AUC_LG_oversampling = numeric(10)
for(i in 1:10){
  cat(i)
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    up_train = upSample(x = train[-testIndexes,-ncol(train)],y = train$TARGET[-testIndexes])
    pred.model = glm( Class ~  . , data = up_train, family = "binomial")
    buffer = predict(pred.model , train[testIndexes,] , type = "response")
    CV_AUC_LG_oversampling[i] = ROCplot(buffer , train$TARGET[testIndexes],main ="Logistic Regression")
}
rm(buffer,up_train,pred.model,testIndexes,i)
mean( CV_AUC_LG_oversampling )

#LDA
library(MASS)

# without sampling
CV_AUC_LDA = numeric(10)
for(i in 1:10){
  cat(i)
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    pred.model = lda( TARGET ~  . , data = train[-testIndexes,])
    buffer = predict(pred.model , train[testIndexes,])
    CV_AUC_LDA[i] = ROCplot(buffer$posterior[,2] , train$TARGET[testIndexes],main ="LDA")
}
rm(buffer,testIndexes,pred.model,i)
mean( CV_AUC_LDA)

# with over-sampling
CV_AUC_LDA = numeric(10)
for(i in 1:10){
  cat(i)
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    up_train = upSample(x = train[-testIndexes,-ncol(train)],y = train$TARGET[-testIndexes])
    pred.model = lda( Class ~  . , data = up_train[-testIndexes, ])
    buffer = predict(pred.model , train[testIndexes,] )
    CV_AUC_LDA_oversampling[i] = ROCplot(buffer$posterior[,2] , train$TARGET[testIndexes],main ="LDA")
}
rm(buffer,up_train,pred.model,i)
mean( CV_AUC_LDA_oversampling)

#Random Forest
library(randomForest)
# train
CV_AUC_rf = NULL
for(i in 1:10){
  cat(i)
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    set.seed(1)
    system.time(pred.model <- randomForest(train[-testIndexes,-ncol(train)], train$TARGET[-testIndexes] ))
    buffer = predict( pred.model , newdata = train[ testIndexes,] , type= "prob")
    CV_AUC_rf[i] = ROCplot(buffer[,2] , train$TARGET[testIndexes],main ="Random Forest")
}

mean(CV_AUC_rf)

# with over sampling
CV_AUC_rf_oversampling = NULL
for(i in 1:3){
  cat(i)
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    set.seed(1)
    up_train = upSample(x = train[-testIndexes,-ncol(train)],y = train$TARGET[-testIndexes])
    pred.model = randomForest(Class ~ ., data= up_train )
    buffer = predict( pred.model , newdata = train[ testIndexes,] , type= "prob")
    CV_AUC_rf_oversampling[i] = ROCplot(buffer[,2] , train$TARGET[testIndexes],main ="Random Forest")
}

mean(CV_AUC_rf_oversampling)

##XGBoost

library(xgboost)
#without sampling
CV_AUC_XG = NULL
xg_train = train
xg_train$TARGET <- as.numeric(levels(xg_train$TARGET))[xg_train$TARGET]
for(i in 1:10){
  cat(i)
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    train_xg <- sparse.model.matrix(TARGET ~ ., data = xg_train[-testIndexes,])
    dtrain <- xgb.DMatrix(data=train_xg, label=xg_train$TARGET[-testIndexes])
    watchlist <- list(train_xg=dtrain)
    param <- list( objective = "binary:logistic",
               booster = "gbtree",
               eval_metric = "auc",
               eta = 0.0202048,
               max_depth = 5,
               subsample = 0.6815,
               colsample_bytree = 0.701)
    clf <- xgb.train( params = param,
                  data = dtrain,
                  nrounds = 560,
                  verbose = 1,
                  watchlist = watchlist,
                  maximize = FALSE)
    test_xg <- sparse.model.matrix(TARGET ~ ., data = xg_train[testIndexes,])
    preds <- predict(clf, test_xg)
    CV_AUC_XG[i] = ROCplot(preds , xg_train$TARGET[testIndexes],main ="XGBoost")
    
}
rm(test_xg,param,i,preds,watchlist,dtrain,train_xg,testIndexes,xg_train)
mean(CV_AUC_XG)


#with over sampling
CV_AUC_XG_oversampling = NULL
for(i in 1:10){
  cat(i)
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    up_train = upSample(x = train[-testIndexes,-ncol(train)],y = train$TARGET[-testIndexes])
    up_train$Class <- as.numeric(levels(up_train$Class))[up_train$Class]
    train_xg <- sparse.model.matrix(Class ~ ., data = up_train)
    dtrain <- xgb.DMatrix(data=train_xg, label=up_train$Class)
    watchlist <- list(train_xg=dtrain)
    param <- list( objective = "binary:logistic",
               booster = "gbtree",
               eval_metric = "auc",
               eta = 0.0202048,
               max_depth = 5,
               subsample = 0.6815,
               colsample_bytree = 0.701)
    clf <- xgb.train( params = param,
                  data = dtrain,
                  nrounds = 560,
                  verbose = 1,
                  watchlist = watchlist,
                  maximize = FALSE)
    test_xg <- sparse.model.matrix(TARGET ~ ., data = train[testIndexes,])
    preds <- predict(clf, test_xg)
    CV_AUC_XG_oversampling[i] = ROCplot(preds , train$TARGET[testIndexes],main ="XGBoost")
    
}
rm(test_xg,param,i,preds,watchlist,dtrain,train_xg,testIndexes,up_train)

mean(CV_AUC)
CV_AUC_XG_oversampling


## ANN
library(nnet)
#without sampling
CV_AUC_ANN = NULL
for(i in 1:10){
    cat(i)
    testIndexes <- which(folds==i,arr.ind=TRUE)
    set.seed(1)
    ideal <- class.ind(as.factor(train$TARGET))
    ANN = nnet( x= train[-testIndexes,-ncol(train)], y= ideal[-testIndexes,], size=5,maxit=200, decay=5e-2)
    pred = predict(ANN, train[testIndexes,-ncol(train)], type="raw")
    
    CV_AUC_ANN[i] = ROCplot(pred[,2] , train$TARGET[testIndexes],main ="ANN")
}
rm(pred,ideal,i,testIndexes,ANN)
mean(CV_AUC_ANN)

#with oversampling
CV_AUC_ANN_oversampling = NULL
for(i in 1:10){
    cat(i)
    testIndexes <- which(folds==i,arr.ind=TRUE)
    set.seed(1)
    up_train = upSample(x = train[-testIndexes,-ncol(train)],y = train$TARGET[-testIndexes])
    ideal <- class.ind(as.factor(up_train$Class))
    ANN = nnet( x= up_train[,-ncol(up_train)], y= ideal, size=5,maxit=200, decay=5e-2)
    pred = predict(ANN, train[testIndexes,-ncol(train)], type="raw")
    
    CV_AUC_ANN_oversampling[i] = ROCplot(pred[,2] , train$TARGET[testIndexes],main ="ANN")
}
rm(pred,ideal,up_train,i,testIndexes,ANN)
mean(CV_AUC_ANN_oversampling)

##H2O GBM

library(h2o)
h2o.init(nthreads=-1)
col <- colnames(train)[-ncol(train)]

#without sampling
train.hex <- as.h2o(train, destination_frame = "train.hex")
CV_AUC_h2Ogmb = NULL
for(i in 1:10){
    cat(i)
    testIndexes <- which(folds==i,arr.ind=TRUE)
   h2o.gbm <- h2o.gbm(y = "TARGET", x = col, training_frame = train.hex[-testIndexes,],
                          ntrees = 500, max_depth = 3, min_rows = 2)
    h2o.gbm.pred = h2o.predict(object = h2o.gbm, newdata = train.hex[testIndexes,])
    
    h2o.gbm.pred.df <-as.data.frame(h2o.gbm.pred)
    CV_AUC_h2Ogmb[i] = ROCplot(h2o.gbm.pred.df [,3] , train$TARGET[testIndexes],main ="H2O GBM")
}
rm(h2o.gbm,h2o.gbm.pred,h2o.gbm.pred.df,i,testIndexes,train.hex)
mean(CV_AUC_h2Ogmb)

#with oversampling
CV_AUC_h2Ogmb_oversampling = NULL
test.hex <- as.h2o(train, destination_frame = "test.hex")
for(i in 1:10){
    cat(i)
    testIndexes <- which(folds==i,arr.ind=TRUE)
    up_train = upSample(x = train[-testIndexes,-ncol(train)],y = train$TARGET[-testIndexes])
    train.hex <- as.h2o(up_train, destination_frame = "train.hex")
    h2o.gbm <- h2o.gbm(y = "Class", x = col, training_frame = train.hex,ntrees = 500, max_depth = 3, min_rows = 2)
    h2o.gbm.pred = h2o.predict(object = h2o.gbm, newdata = test.hex[testIndexes,])
    
    h2o.gbm.pred.df <-as.data.frame(h2o.gbm.pred)
    CV_AUC_h2Ogmb_oversampling[i] = ROCplot(h2o.gbm.pred.df [,3] , train$TARGET[testIndexes],main ="H2O GBM")
}

rm(h2o.gbm,h2o.gbm.pred,h2o.gbm.pred.df,i,testIndexes,train.hex,test.hex,up_train)
mean(CV_AUC_h2Ogmb_oversampling)



