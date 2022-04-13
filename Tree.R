#load Data
library(MASS) #this data is in MASS package
boston_data <- data(Boston) 
sample_index <- sample(nrow(Boston),nrow(Boston)*0.90) 
boston_train <- Boston[sample_index,] 
boston_test <- Boston[-sample_index,]

install.packages('rpart') 
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)
boston_rpart <- rpart(formula = medv ~ ., data = boston_train)

###### Printing and ploting the tree ######
boston_rpart
prp(boston_rpart,digits = 4, extra = 1)

##### Prediction using regression trees #####
### In-sample prediction #####
boston_train_pred_tree = predict(boston_rpart)
mean((boston_train_pred_tree - boston_train$medv)^2) 
### Out-of-sample prediction #####
boston_test_pred_tree = predict(boston_rpart,boston_test)
mean((boston_test_pred_tree - boston_test$medv)^2)

#Linear Regression Model
boston.reg = lm(medv~., data = boston_train) 
#Insample
boston_train_pred_reg = predict(boston.reg, boston_train) 
mean((boston_train_pred_reg - boston_train$medv)^2)
#out-of-sample
boston_test_pred_reg = predict(boston.reg, boston_test) 
mean((boston_test_pred_reg - boston_test$medv)^2)

### Pruning #####
boston_largetree <- rpart(formula = medv ~ ., data = boston_train, cp =0.001)
prp(boston_largetree)
plotcp(boston_largetree)
printcp(boston_largetree)

sum((boston_train$medv - mean(boston_train$medv))^2)/nrow(boston_train)
mean((predict(boston_largetree) - boston_train$medv)^2)

treepruned=prune(boston_largetree, cp = 0.0095)
prp(treepruned)

boston_final <- rpart(formula = medv ~ ., data = boston_train, cp =0.0095)

##############   Start of  ################
########## Classification Tree ##############
#############################################

library(rpart)
library(rpart.plot)

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

#Fitting the Model
german_credit_rpart<- rpart(formula = response~.,data=german_credit_train, method = 'class', parms = list(loss = matrix(c(0,5,1,0), nrow=2)))
#printing and plotting
print(german_credit_rpart)
prp(german_credit_rpart, extra = 1)

#1)In-sample-prediction
german_credit_train.pred.tree_insample<- predict(german_credit_rpart, german_credit_train, type="class")
#Misclassification Rate
table(german_credit_train$response, german_credit_train.pred.tree_insample, dnn=c("Truth", "Predicted"))

#2)Out-of-sample prediction
german_credit_test.pred.tree_insample<- predict(german_credit_rpart, german_credit_test, type="class")
#Misclassification Rate
table(german_credit_test$response, german_credit_test.pred.tree_insample, dnn=c("Truth", "Predicted"))

#Cost
cost <-function(r, phat){ 
  weight1 <-5
  weight0 <-1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
} 

## Calculate the actual cost using a self-defined cost function.
cost(german_credit_train$response, predict(german_credit_rpart, german_credit_train, type="prob"))
cost(german_credit_test$response, predict(german_credit_rpart, german_credit_test, type="prob"))

##########Start of Comparing with ############
##########Logistic Regression ##################
#############################################
#Fit logistic regression model
german_credit_glm <- glm(response~., data = german_credit_train, family=binomial)
#Get binary prediction
german_credit_test_pred_glm <- predict(german_credit_glm, german_credit_test, type="response")
#Calculate cost using test set
cost(german_credit_test$response, german_credit_test_pred_glm)
table(german_credit_test$response, as.numeric(german_credit_test_pred_glm>1/6), dnn=c("Truth", "Predicted"))

##########End of Comparing with ############
##########Logistic Regression ##################
###### ROC Curve and Cut-offProbability ######
#Probability of getting 1
german_credit_test_prob_rpart = predict(german_credit_rpart, german_credit_test, type="prob")

library(ROCR)
pred = prediction(german_credit_test_prob_rpart[,2], german_credit_test$response) 
perf = performance(pred,"tpr","fpr") 
plot(perf, colorize=TRUE)#slot(performance(pred,"auc"),"y.values")[[1]]#
german_credit_test_pred_rpart = as.numeric(german_credit_test_prob_rpart[,2] >1/(5+1)) 
table(german_credit_test$response, german_credit_test_pred_rpart, dnn=c("Truth", "Predicted"))

##############   End of  ################
########## Classification Tree ##############











