library(neuralnet)
# load Boston data
library(MASS)
data(Boston)
index <- sample(nrow(Boston),nrow(Boston)*0.90)
boston.train <- Boston[index,]
boston.test <- Boston[-index,]

####### Note, the response variable has to be rescaled to (0,1) range, 
####### and predictors are recommended to be standardized as well.
#this part is to format data set to fit neural network
boston.train[,-14]<- as.data.frame(scale(boston.train[,-14])) #training data set
boston.train$medv<- boston.train$medv/50 #training independent variable
boston.test[,-14]<- as.data.frame(scale(boston.test[,-14])) #testing data set
boston.test$medv<- boston.test$medv/50 #training independent variable
# fit neural network with one hidden layer
boston.ann1<- neuralnet(medv~., data = boston.train, hidden = 5, linear.output = TRUE)
#Plot the neural network
plot(boston.ann1)
####### Prediction on training sample.
boston.pred1<- compute(boston.ann1, boston.train)
boston.pred1<- boston.pred1$net.result*50
mean((boston.train$medv-boston.pred1)^2)
#testing
boston.pred2<- compute(boston.ann1, boston.test)
boston.pred2<- boston.pred2$net.result*50
mean((boston.test$medv-boston.pred2)^2)
#cart
library(rpart)
library(rpart.plot)
boston_rpart <- rpart(formula = medv ~ ., data = boston_train)

###### Printing and ploting the tree ######
boston_rpart
prp(boston_rpart,digits = 4, extra = 1)
### In-sample prediction #####
boston_train_pred_tree = predict(boston_rpart)
mean((boston_train_pred_tree - boston.train$medv)^2) 
### Out-of-sample prediction #####
boston_test_pred_tree = predict(boston_rpart,boston_test)
mean((boston_test_pred_tree - boston.test$medv)^2)


###########################German-Credit#########################
#################################################
library(neuralnet)

# load credit card data
german_credit = read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", header = T)
colnames(german_credit) = c("chk_acct", "duration", "credit_his",
                            "purpose", "amount", "saving_acct", "present_emp", "installment_rate",
                            "sex", "other_debtor", "present_resid", "property", "age",
                            "other_install", "housing", "n_credits", "job", "n_people",
                            "telephone", "foreign", "response")
german_credit$response = german_credit$response - 1
german_credit$response <- as.factor(german_credit$response)
str(german_credit)
#Data Cleaning
german_credit$chk_acct<- as.factor(german_credit$chk_acct) 
german_credit$credit_his<- as.factor(german_credit$credit_his) 
german_credit$purpose<- as.factor(german_credit$purpose)
german_credit$saving_acct<- as.factor(german_credit$saving_acct)
german_credit$present_emp<- as.factor(german_credit$present_emp)
german_credit$sex<- as.factor(german_credit$sex)
german_credit$ other_debtor<- as.factor(german_credit$ other_debtor)
german_credit$property<- as.factor(german_credit$property)
german_credit$other_install<- as.factor(german_credit$other_install)
german_credit$housing<- as.factor(german_credit$housing)
german_credit$job<- as.factor(german_credit$job)
german_credit$telephone<- as.factor(german_credit$telephone)
german_credit$foreign<- as.factor(german_credit$foreign)
str(german_credit)

#Train and Test
index <- sample(nrow(german_credit),nrow(german_credit)*0.80) 
german_credit_train = german_credit[index,] 
german_credit_test = german_credit[-index,]
####### First standardize the data and create dummy variables for categorical data.

credit.train.X<- as.data.frame(scale(model.matrix(~., data = german_credit_train[,-ncol(german_credit_train)])[,-1]))
credit.train.Y<- german_credit_train[,ncol(german_credit_train)]
credit.train1<- data.frame(response= credit.train.Y, credit.train.X)
credit.test1<- as.data.frame(scale(model.matrix(~., data = german_credit_test[,-ncol(german_credit_train)])[,-1]))
# fit neural networks. CAUTION: this is slow
credit.ann<- neuralnet(response~., data = credit.train1, hidden = 3, linear.output = FALSE)
plot(credit.ann)

####### Prediction on training sample.
credit.pred1<- neuralnet::compute(credit.ann, credit.test1)
head(cbind(german_credit_train$response, credit.pred1$net.result), 10)
detach(package:neuralnet) 
####### ROC curve and AUC
library(ROCR)
pred <- prediction(credit.pred1$net.result, german_credit_train$response)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

####### Confusion matrix
credit.pred01.ann<- (credit.pred1$net.result>mean(german_credit_train$response))*1
#Confusion matrix
table(german_credit_train$response, credit.pred01.ann, dnn=c("Truth","Predicted"))








