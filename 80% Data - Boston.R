library(MASS)
data(Boston); #this data is in MASS package
colnames(Boston)
#subset
subset <- sample(nrow(Boston), nrow(Boston) * 0.8)
Boston_train = Boston[subset, ]
Boston_test = Boston[-subset, ]
#exploratoryanalysis
boxplot(Boston_train)
summary(Boston_train)
#linearregression
model_1 <- lm(medv ~ ., data = Boston_train)
summary(model_1)

model_summary <- summary(model_1)
(model_summary$sigma)^2
model_summary$adj.r.squared
BIC(model_1)
AIC(model_1)
#variableselection
install.packages("leaps")
library(leaps)

subset_result <- regsubsets(medv ~ ., data = Boston_train, nbest = 2, nvmax = 14)
summary(subset_result)

nullmodel = lm(medv ~ 1, data = Boston_train)
fullmodel = lm(medv ~ ., data = Boston_train)

model.step <- step(fullmodel, direction = "backward")
summary(model.step)

model_summary <- summary(model.step)
(model_summary$sigma)^2
model_summary$adj.r.squared

plot(model.step$fitted.values , model.step$residuals)

#out-of-sample
model_3 <- lm(medv ~ lstat + rm + ptratio + black + dis + nox + chas + zn + 
                rad + tax + crim, data = Boston_test)

pi <- predict(object = model_3, newdata = Boston_test)
pi <- predict(model_3, Boston_test)
mean((pi - Boston_test$medv)^2)


model_2 = glm(medv ~ indus + rm, data = Boston)
cv.glm(data = Boston, glmfit = model_2, K = 10)$delta[2]

model_2 = glm(medv ~ indus + rm, data = Boston)

MAE_cost = function(pi, r) {
  return(mean(abs(pi - r)))
}
cv.glm(data = Boston, glmfit = model_2, cost = MAE_cost, K = 10)$delta[2]
