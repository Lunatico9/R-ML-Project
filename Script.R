#use readr to import dataset
if (!require(readr)) install.packages('readr')
library(readr)
#read dataset
dataset = read_csv('./column_3C_weka.csv')
#make class a factor variable
dataset$class = as.factor(dataset$class)

#load required packages
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(gtools)) install.packages('caret')
library(gtools)
if (!require(caTools)) install.packages('caTools')
library(caTools)
if (!require(rpart)) install.packages('rpart')
library(rpart)
if (!require(randomForest)) install.packages('randomForest')
library(randomForest)
if (!require(kernlab)) install.packages('kernlab')
library(kernlab)
##Package for MLP implementation
#if (!require(RSNNS)) install.packages('RSNNS')
#library(RSNNS)
#split dataset into train and test sets
index = createDataPartition(dataset$class, p = 0.7, list = FALSE)
train = dataset[index,]
test = dataset[-index,]

#fit boosted logistic regression model, a logistic variant of AdaBoost
fit_logitBoost = caret::train(class ~ ., data = train, method = "LogitBoost")

#fit a classification decision tree
fit_CART = caret::train(class ~ ., data = train, method = "rpart")

#fit a knn model
fit_knn = caret::train(class ~ ., data = train, method = "knn", tuneGrid = data.frame(k = seq(3,33,2)))

#fit a random forest model
fit_rf = caret::train(class ~ ., data = train, method = "rf")

######################
##MLP training and prediction. Ineffective and slow (accuracy around 45%!).
######################
##define layers as all permutations of numbers from 1 to 5 (1 - number of predictors)
#layers = data.frame(permutations(n = 5, r = 3, v = 1:5))
#colnames(layers) = c("layer1","layer2","layer3")
##filter out neural networks that feed forward to a larger layer (decode information, harmful here)
#layers = layers %>% filter(layer1 >= layer2 & layer2 >= layer3)
##fit neural network
#fit_mlp = train(class ~ ., data = train, method = "mlpML", tuneGrid = layers)
# y_mlp = predict(fit_mlp, newdata = test)
# confusionMatrix(y_mlp, reference = test$class)
######################

#fit support vector machine model
fit_svm = caret::train(class ~ ., data = train, method = "svmLinear", tuneGrid = data.frame(C = seq(0.01,10,1)))

#predict using logitBoost
y_logit = predict(fit_logitBoost, newdata = test)
caret::confusionMatrix(y_logit, reference = test$class)

#predict using classification tree
y_CART = predict(fit_CART, newdata = test)
#display variable importance
fit_CART$finalModel$variable.importance
caret::confusionMatrix(y_CART, reference = test$class)

#predict using knn
y_knn = predict(fit_knn, newdata = test)
caret::confusionMatrix(y_knn, reference = test$class)

#predict using random forest
y_rf = predict(fit_rf, newdata = test)
#display variable importance
fit_rf$finalModel$importance
caret::confusionMatrix(y_rf, reference = test$class)

#predict using support vector machine
y_svm = predict(fit_svm, newdata = test)
caret::confusionMatrix(y_svm, reference = test$class)

#this function returns a vector containing the accuracy of the machine learning algorithms
accuracies = function(dataset){
  index = createDataPartition(dataset$class, p = 0.7, list = FALSE)
  
  train = dataset[index,]
  test = dataset[-index,]
  
  fit_logitBoost = caret::train(class ~ ., data = train, method = "LogitBoost")
  fit_CART = caret::train(class ~ ., data = train, method = "rpart")
  fit_knn = caret::train(class ~ ., data = train, method = "knn", tuneGrid = data.frame(k = seq(3,33,2)))
  fit_rf = caret::train(class ~ ., data = train, method = "rf")
  fit_svm = caret::train(class ~ ., data = train, method = "svmLinear")
  
  y_logit = predict(fit_logitBoost, newdata = test)
  y_CART = predict(fit_CART, newdata = test)
  y_knn = predict(fit_knn, newdata = test)
  y_rf = predict(fit_rf, newdata = test)
  y_svm = predict(fit_svm, newdata = test)
  c(
    caret::confusionMatrix(y_logit, reference = test$class)$overall[[1]],
    caret::confusionMatrix(y_CART, reference = test$class)$overall[[1]],
    caret::confusionMatrix(y_knn, reference = test$class)$overall[[1]],
    caret::confusionMatrix(y_rf, reference = test$class)$overall[[1]],
    caret::confusionMatrix(y_svm, reference = test$class)$overall[[1]]
  )
}

#WARNING: takes a while to run
#calculate aggregated accuracy over 100 random train/test splits
aggregated_acc = replicate(100, accuracies(dataset))

# transpose and assign names
aggregated_acc = t(aggregated_acc)
colnames(aggregated_acc) = c("LogitBoost", "Classification Tree", "K-NN", "Random Forest", "Support Vector Machine")
#report mean accuracy for each model
mean_acc = t(as.matrix(colMeans(aggregated_acc)))
rownames(mean_acc) = c("Mean accuracy")
mean_acc
#report standard deviation of accuracy for each model
acc_sd = t(as.matrix(c(sd(aggregated_acc[,1]), sd(aggregated_acc[,2]), sd(aggregated_acc[,3]), sd(aggregated_acc[,4]), sd(aggregated_acc[,5]))))
colnames(acc_sd) = c("LogitBoost", "Classification Tree", "K-NN", "Random Forest", "Support Vector Machine")
rownames(acc_sd) = c("Standard deviation of accuracy")
acc_sd