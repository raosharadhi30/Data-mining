#Clustering Analysis

install.packages("default")
library(default)
data("iris")
index <- sample(nrow(iris),nrow(iris)*0.90)
iris.train <- iris[index,]
iris.test <- iris[-index,]

install.packages("fpc")
library(fpc)# K-Means Cluster Analysis
fit <- kmeans(iris.train[,1:4], 3)#3 cluster solution
# Display number of clusters in each cluster
table(fit$cluster)
plotcluster(iris.train[,1:4], fit$cluster ) #plot
#to see which items are in first group
iris.train$Species[fit$cluster == 1] 
iris.train$Species[fit$cluster == 2] 
iris.train$Species[fit$cluster == 3] 
# get cluster means 
aggregate(iris.train[,1:4], by = list(fit$cluster), FUN = mean)

fit <- kmeans(iris.train[,1:4], 2)#2 cluster solution
table(fit$cluster)
plotcluster(iris.train[,1:4], fit$cluster ) #plot
#to see which items are in first group
iris.train$Species[fit$cluster == 1] 
iris.train$Species[fit$cluster == 2] 
aggregate(iris.train[,1:4], by = list(fit$cluster), FUN = mean)

fit <- kmeans(iris.train[,1:4], 5)#5 cluster solution
table(fit$cluster)
plotcluster(iris.train[,1:4], fit$cluster ) #plot
#to see which items are in first group
iris.train$Species[fit$cluster == 1] 
iris.train$Species[fit$cluster == 2] 
iris.train$Species[fit$cluster == 3] 
iris.train$Species[fit$cluster == 4] 
iris.train$Species[fit$cluster == 5] 
aggregate(iris.train[,1:4], by = list(fit$cluster), FUN = mean)

# Wards Method or Hierarchical clustering Calculate the distance matrix
Iris.train.dist = dist(iris.train[,1:4])# Obtain clusters using the Wards method
Iris.hclust = hclust(Iris.train.dist, method ="ward.D") 
#win.graph()
plot(Iris.hclust)


#####Hirarchial############## 
# Cut dendrogram at the 3 clusters level and obtain cluster membership
groupIris.3 = cutree(Iris.hclust, k =3)
table(groupIris.3)
# See exactly which item are in third group
iris$Species[groupIris.3 ==3]#change the no from 1 to 3 based on what item you want to look
aggregate(iris.train[,1:4], by = list(groupIris.3), FUN = mean)
# Centroid Plot 
library(fpc)
plotcluster(iris.train[,1:4], groupIris.3)


# Cut dendrogram at the 2 clusters level and obtain cluster membership
groupIris.2 = cutree(Iris.hclust, k =2)
table(groupIris.2)
# See exactly which item are in second group
iris$Species[groupIris.2 ==1]#change the no from 1 to 2 based on what item you want to look
aggregate(iris.train[,1:4], by = list(groupIris.2), FUN = mean)
# Centroid Plot 
library(fpc)
plotcluster(iris.train[,1:4], groupIris.2)


# Cut dendrogram at the 5 clusters level and obtain cluster membership
groupIris.5 = cutree(Iris.hclust, k =5)
table(groupIris.5)
# See exactly which item are in fifth group
iris$Species[groupIris.5 ==1] #change the no from 1 to 5 based on what item you want to look
aggregate(iris.train[,1:4], by = list(groupIris.5), FUN = mean)
# Centroid Plot 
library(fpc)
plotcluster(iris.train[,1:4], groupIris.5)

#########################
####Association Rules####
#########################

#preliminary
install.packages("arules")
library(arules)
data(Groceries)
summary(Groceries) #most frequent items and also average

#data exploration
dim(Groceries) #dimension of grocery data set
inspect(Groceries[1:10]) #Print out the first 10 transactions.
itemFrequencyPlot(Groceries) #freq plot
itemFrequencyPlot(Groceries, support= 0.1, cex.names= 0.8) #freq plot with atleast 10% support

#associate rules
rules<-apriori(Groceries)
## Over ride the default minimum support and confidence
rules<-apriori(Groceries, parameter= list(support= 0.01, confidence= 0.4))
rules ## Print out the number of rules
## A summary of the rules generated
summary(rules) #distribution of number of item in each rule and average of lift
inspect(rules)## Use inspect to see the rules
inspect(head(sort(rules, by= "lift"), n= 5)) #to Print out the five rules with the highest lift.



