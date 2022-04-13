library(MASS)
data(USJudgeRatings)
summary(USJudgeRatings)
USJudgeRatings

# Sampling into Training and Testing data
test <- sample(nrow(USJudgeRatings), nrow(USJudgeRatings)*0.9)
USJudgeRatings_train <- USJudgeRatings[test,]
USJudgeRatings_test <- USJudgeRatings[-test,]

summary(USJudgeRatings_train)
dim(USJudgeRatings_train)

#EDA
#Scatter Plot
plot(USJudgeRatings_train)

#Boxplot
par(mfrow=c(1,1))
boxplot(USJudgeRatings_train)

# Histograms
par(mfrow=c(3,4))
hist(USJudgeRatings_train$CONT,
     main = "Number of contacts of lawyer with judge")
hist(USJudgeRatings_train$INTG,
     main = "Judicial integrity")
hist(USJudgeRatings_train$DMNR,
     main = "Demeanor")
hist(USJudgeRatings_train$DILG,
     main = "Diligence")
hist(USJudgeRatings_train$CFMG,
     main = "Case flow managing")
hist(USJudgeRatings_train$DECI, 
     main = "Prompt decisions")
hist(USJudgeRatings_train$PREP, 
     main = "Preparation for trial")
hist(USJudgeRatings_train$FAMI,  
     main = "Familiarity with law")
hist(USJudgeRatings_train$ORAL, 
     main = "Sound oral rulings")
hist(USJudgeRatings_train$WRIT, 
     main = "Sound written rulings")
hist(USJudgeRatings_train$PHYS, 
     main = "Physical ability")
hist(USJudgeRatings_train$RTEN, 
     main = "Worthy of retention")
# Density Curves
par(mfrow=c(3,4))
plot(density(USJudgeRatings_train$CONT),
     main = "Number of contacts of lawyer with judge")
plot(density(USJudgeRatings_train$INTG),
     main = "Judicial integrity")
plot(density(USJudgeRatings_train$DMNR),
     main = "Demeanor")
plot(density(USJudgeRatings_train$DILG),
     main = "Diligence")
plot(density(USJudgeRatings_train$CFMG),
     main = "Case flow managing")
plot(density(USJudgeRatings_train$DECI), 
     main = "Prompt decisions")
plot(density(USJudgeRatings_train$PREP), 
     main = "Preparation for trial")
plot(density(USJudgeRatings_train$FAMI),  
     main = "Familiarity with law")
plot(density(USJudgeRatings_train$ORAL), 
     main = "Sound oral rulings")
plot(density(USJudgeRatings_train$WRIT), 
     main = "Sound written rulings")
plot(density(USJudgeRatings_train$PHYS), 
     main = "Physical ability")
plot(density(USJudgeRatings_train$RTEN), 
     main = "Worthy of retention")

#  Step 4 - Fitting Linear regression:
USJudgeRatings_lm <- lm(RTEN~., data=USJudgeRatings_train)

#Summary on Linear regression
summary(USJudgeRatings_lm)
summ_lm <- summary(USJudgeRatings_lm)
par(mfrow=c(2,2))
plot(USJudgeRatings_lm)

# Model Evaluation
(summ_lm$sigma)^2 #MSE
AIC(USJudgeRatings_lm)
BIC(USJudgeRatings_lm)

#In-sample-prediction
judge_train_linmod = predict(USJudgeRatings_lm)
mean((judge_train_linmod - USJudgeRatings_train$RTEN)^2) #MSE
mean(abs(resid-USJudgeRatings_train$RTEN)) #MAE

#Out-of-sample prediction
judge_test_linmod = predict(USJudgeRatings_lm, USJudgeRatings_test)
resid = judge_test_linmod - USJudgeRatings_test$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_test$RTEN)) #MAE

# Variable Selection

#Best Subset Regression
#install.packages("leaps")
library(leaps)

subset_result <- regsubsets (RTEN ~., data = USJudgeRatings_train, nbest=5, method = "exhaustive")
summary(subset_result)
par(mfrow=c(1,1))
plot(subset_result,scale='bic')
plot(subset_result,scale='Cp')
plot(subset_result,scale='adjr2')
plot(subset_result,scale='r2')


#Backward Elimination using AIC
nullmodel = lm(RTEN ~ 1, data = USJudgeRatings_train)
fullmodel = lm(RTEN ~. , data = USJudgeRatings_train)

back_AIC <- step(fullmodel)
summary(back_AIC) 

#In-sample-prediction
judge_train_linmod = predict(back_AIC)
resid = judge_train_linmod - USJudgeRatings_train$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_train$RTEN)) #MAE

#Out-of-sample prediction
judge_test_linmod = predict(back_AIC, USJudgeRatings_test)
resid = judge_test_linmod - USJudgeRatings_test$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_test$RTEN)) #MAE


#Backward Elimination using BIC
back_BIC <- step(USJudgeRatings_lm, k=log(nrow(USJudgeRatings_train)))
summary(back_BIC) 

#In-sample-prediction
judge_train_linmod = predict(back_BIC)
resid = judge_train_linmod - USJudgeRatings_train$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_train$RTEN)) #MAE

#Out-of-sample prediction
judge_test_linmod = predict(back_BIC, USJudgeRatings_test)
resid = judge_test_linmod - USJudgeRatings_test$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_test$RTEN)) #MAE


#Forward Selection using AIC
forward_AIC <- step(nullmodel, scope = list(lower=nullmodel, upper=fullmodel), direction = "forward")
summary(forward_AIC) 


#In-sample-prediction
judge_train_linmod = predict(forward_AIC)
resid = judge_train_linmod - USJudgeRatings_train$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_train$RTEN)) #MAE

#Out-of-sample prediction
judge_test_linmod = predict(forward_AIC, USJudgeRatings_test)
resid = judge_test_linmod - USJudgeRatings_test$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_test$RTEN)) #MAE


#### Forward selection with BIC ######
forward_BIC <- step(nullmodel, scope = list(lower=nullmodel,upper=fullmodel), direction="forward", criterion = "BIC")
summary(forward_BIC) 

#In-sample-prediction
judge_train_linmod = predict(forward_BIC)
resid = judge_train_linmod - USJudgeRatings_train$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_train$RTEN)) #MAE

#Out-of-sample prediction
judge_test_linmod = predict(forward_BIC, USJudgeRatings_test)
resid = judge_test_linmod - USJudgeRatings_test$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_test$RTEN)) #MAE



####### Stepwise selection both #######
both_step <- step(nullmodel, scope = list(lower = nullmodel, upper = fullmodel), direction = "both")
summary(both_step) 

#In-sample-prediction
judge_train_linmod = predict(both_step)
resid = judge_train_linmod - USJudgeRatings_train$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_train$RTEN)) #MAE

#Out-of-sample prediction
judge_test_linmod = predict(both_step, USJudgeRatings_test)
resid = judge_test_linmod - USJudgeRatings_test$RTEN
mean(resid^2) #MSE
mean(abs(resid-USJudgeRatings_test$RTEN)) #MAE


############## TREE ###################
### Classification Tree ####


library(rpart)
library(rpart.plot)

#Model - Fitting
USJudgeRatings_tree <- rpart(formula =RTEN ~., data = USJudgeRatings_train, cp = 0.000001)

#printing and plotting
USJudgeRatings_tree
prp(USJudgeRatings_tree,digits =4, extra =1)

prp(USJudgeRatings_tree)
plotcp(USJudgeRatings_tree)
printcp(USJudgeRatings_tree)
treepruned = prune(USJudgeRatings_tree, cp = 0.004)
prp(treepruned)

##### Prediction using regression trees #####

### In-sample prediction #####
USJudgeRatings_train_tree = predict(USJudgeRatings_tree)
resid = USJudgeRatings_train_tree - USJudgeRatings_train$RTEN #finding residuals
mean(resid^2) #MSE

### Out-of-sample prediction #####
USJudgeRatings_test_tree = predict(USJudgeRatings_tree,USJudgeRatings_test)
resid = USJudgeRatings_test_tree - USJudgeRatings_test$RTEN #finding residuals
mean(resid^2) #MSE


############## NEURAL NETWORK ###################

library(neuralnet)

USJudgeRatings_train[,-12]<- as.data.frame(scale(USJudgeRatings_train[,-12]))
USJudgeRatings_train$RTEN<- USJudgeRatings_train$RTEN/9
USJudgeRatings_test[,-12]<- as.data.frame(scale(USJudgeRatings_test[,-12]))

# fit neural network with one hidden layer
USJudgeRatings.ann<- neuralnet(RTEN ~., data = USJudgeRatings_train, hidden = 2, linear.output = TRUE)
#Plot the neural network
plot(USJudgeRatings.ann)



####### Prediction in sample.

USJudgeRatings_isp <- compute(USJudgeRatings.ann, USJudgeRatings_train)
USJudgeRatings.pred1<- USJudgeRatings_isp$net.result*9
mean((USJudgeRatings_train$RTEN-USJudgeRatings.pred1)^2)

####### Prediction out of sample.

USJudgeRatings_oosp <- compute(USJudgeRatings.ann, USJudgeRatings_test)
USJudgeRatings.pred2<- USJudgeRatings_oosp$net.result*9
mean((USJudgeRatings_test$RTEN-USJudgeRatings.pred2)^2)



################# BOOSTING ###################

install.packages("gbm")
library(gbm)
set.seed(1)
USJudgeRatings_boost = gbm(RTEN~., data=USJudgeRatings_train, distribution= "gaussian",n.trees=500, interaction.depth=2)
summary(USJudgeRatings_boost)


yhat.boost=predict(USJudgeRatings_boost,newdata=USJudgeRatings_train, n.trees=1000)
resid = yhat.boost - USJudgeRatings_train$RTEN #finding residuals
mean(resid^2) #MSE


yhat.boost=predict(USJudgeRatings_boost,newdata=USJudgeRatings_test, n.trees=1000)
resid = yhat.boost - USJudgeRatings_test$RTEN #finding residuals
mean(resid^2) #MSE


plot(USJudgeRatings_boost ,i="perf")
plot(USJudgeRatings_boost ,i="syct")

