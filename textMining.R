library(tm)
library(SnowballC)
library(wordcloud)
library(topicmodels)

## load the data ####
reviews = read.csv("desktop/movie_reviews.csv", stringsAsFactors = F, row.names = 1)
#90%train
index <- sample(nrow(reviews),nrow(reviews)*0.90)
ReviewSample = reviews[index,] 
ReviewsSample_test = reviews[-index,]

##### data prepration 
review_corpus = Corpus(VectorSource(ReviewSample$content))#Bag of words approach / Converting each row into a list of words
review_corpus = tm_map(review_corpus, content_transformer(tolower))#Converting Uppercase to lowercase
review_corpus = tm_map(review_corpus, removeNumbers)#Removing Numbers
review_corpus = tm_map(review_corpus, removePunctuation)#Removing Punctuation
review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))#Removing stopwords
review_corpus =  tm_map(review_corpus, stripWhitespace)#Removing Whitespaces

## check
inspect(review_corpus[1])
#1) Use the sample data to draw word clouds using frequencies of the termsand TF-IDF (term frequency-inverse document frequency. 
#   Please comment on those word clouds. Which one is more informative?

#Analyzing the textual data using Document-Term Matrix (DTM)
review_dtm <- DocumentTermMatrix(review_corpus)
review_dtm
#check
inspect(review_dtm[500:505, 500:505])
inspect(review_dtm[1,1:20])
findFreqTerms(review_dtm, 1000)
freq = data.frame(sort(colSums(as.matrix(review_dtm)), decreasing=TRUE))

### data visualization
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))
#Using Tf-Idf to improve the model
review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
review_dtm_tfidf
inspect(review_dtm_tfidf[1,1:20])
freq = data.frame(sort(colSums(as.matrix(review_dtm_tfidf)), decreasing=TRUE))
### data visualization
wordcloud(rownames(freq), freq[,1], max.words=100, colors=brewer.pal(1, "Dark2"))

#2)Calculate the number of positive words and negative words in each document using a precompiled 
# list of words with positive and negative meanings. Provide a brief summary on those sentiment variables.
#Sentiment Analysis

neg_words = read.table("Desktop/negative-words.txt", header = F, stringsAsFactors = F)[, 1]
pos_words = read.table("Desktop/positive-words.txt", header = F, stringsAsFactors = F)[, 1]

### Matching 
ReviewSample$neg = tm_term_score(DocumentTermMatrix(review_corpus), neg_words)
ReviewSample$pos = tm_term_score(DocumentTermMatrix(review_corpus), pos_words)

ReviewSample$content = NULL 

ReviewSample = cbind(ReviewSample, as.matrix(review_dtm_tfidf))
ReviewSample$polarity = as.factor(ReviewSample$polarity)

#3)

id_train <- sample(nrow(ReviewSample),nrow(ReviewSample)*0.80)
ReviewSample.train = ReviewSample[id_train,]
ReviewSample.test = ReviewSample[-id_train,]

###Fitting logistic regression model
library(ggplot2)
library(dplyr)

reviews.glm = glm(polarity~ ., family = "binomial", data =ReviewSample.train); 
summary(head(reviews.glm))
#insample
pred_resp <- predict(reviews.glm,type="response") 
hist(pred_resp)
#missclassification matrix
table(ReviewSample.train$polarity, (pred_resp>0.5)*1, dnn=c("Truth","Predicted"))
#MCR
mean(ifelse(ReviewSample.train$polarity != pred_resp,1,0))
#Cost
cost1 <- function(r, pi, pcut){
  mean(((r==0)&(pi>pcut)) | ((r==1)&(pi<pcut)))
}

cost2 <-function(r, pi, pcut){ 
  weight1 <-5
  weight0 <-1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
}

pcut <-1/(5+1)

#Symmetric cost
cost1(r = ReviewSample.train$polarity, pi = pred_resp, pcut)
#Asymmetric cost
cost2(r = ReviewSample.train$polarity, pi = pred_resp, pcut)

#ROC Curve
library(ROCR)
pred = prediction(pred_resp, ReviewSample.train$polarity)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

#Out-of-sample prediction - Logistic
#Prediction
pred_resp_test = predict(reviews.glm, newdata = ReviewSample.test, type = 'response')
#Confusion Matrix/ mis-classification rate
table(ReviewSample.test$polarity, (pred_resp_test >0.5)*1, dnn=c("Truth","Predicted"))

#MCR
mean(ifelse(ReviewSample.test$polarity != pred_resp_test,1,0))

#Costs
cost1 <- function(r, pi, pcut){
  mean(((r==0)&(pi>pcut)) | ((r==1)&(pi<pcut)))
}

cost2 <-function(r, pi, pcut){ 
  weight1 <-5
  weight0 <-1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
}

pcut <-1/(5+1)
#Symmetric cost
cost1(r = ReviewSample.test$polarity, pi = pred_resp_test, pcut)

#Asymmetric cost
cost2(r = ReviewSample.test$polarity, pi = pred_resp_test, pcut)

#ROC Curve
pred = prediction(pred_resp_test, ReviewSample.test$polarity)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

####Fitting Classification Trees
library(rpart)
library(rpart.plot)
library(e1071)
library(nnet)
reviews.tree = rpart(polarity~., method = 'class', data = ReviewSample.train)
prp(reviews.tree)

#In-Sample prediction - Classification Trees
pred_tree<- predict(reviews.tree, ReviewSample.train, type="class")

#Confusion Matrix
table(ReviewSample.train$polarity, pred_tree, dnn=c("Truth", "Predicted"))

#MCR
mean(ifelse(ReviewSample.train1$polarity != pred_tree,1,0))

#Cost
cost1 <-function(r, phat){
  mean(((r==0)&(phat>pcut)) | ((r==1)&(phat<pcut)))
}

cost2 <-function(r, phat){ 
  weight1 <-5
  weight0 <-1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
}


#Actual cost Symmetric
cost1(ReviewSample.train$polarity, predict(reviews.tree, ReviewSample.train, type="prob"))

#Actual cost ASymmetric
cost2(ReviewSample.train$polarity, predict(reviews.tree, ReviewSample.train, type="prob"))

#Probability of getting 1
reviews.train1_prob_rpart = predict(reviews.tree, ReviewSample.train, type="prob")
#ROC Curve
library(ROCR)
pred = prediction(reviews.train1_prob_rpart[,2], ReviewSample.train$polarity)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

#Out-of-Sample prediction - Classification Trees
pred_tree_test<- predict(reviews.tree, ReviewSample.test, type="class")

#Confusion Matrix
table(ReviewSample.test$polarity, pred_tree_test, dnn=c("Truth", "Predicted"))

#MCR
mean(ifelse(ReviewSample.test$polarity != pred_tree_test,1,0))

#Cost
cost1 <-function(r, phat){
  mean(((r==0)&(phat>pcut)) | ((r==1)&(phat<pcut)))
}

cost2 <-function(r, phat){ 
  weight1 <-5
  weight0 <-1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
}

#Actual cost Symmetric
cost1(ReviewSample.test$polarity, predict(reviews.tree, ReviewSample.test, type="prob"))

#Actual cost ASymmetric
cost2(ReviewSample.test$polarity, predict(reviews.tree, ReviewSample.test, type="prob"))

#Probability of getting 1
reviews.test1_prob_rpart = predict(reviews.tree, ReviewSample.test, type="prob")
#ROC Curve
library(ROCR)
pred = prediction(reviews.test1_prob_rpart[,2], ReviewSample.test$polarity)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

###Fitting Neural Networks
library(nnet)
library(NeuralNetTools)
reviews.nnet<- nnet(polarity~., data = ReviewSample.train, size = 1, maxit = 500)
plotnet(reviews.nnet, ynames = 'polarity')

#In-Sample
prob.nnet= predict(reviews.nnet,ReviewSample.train)
pred.nnet = as.numeric(prob.nnet > 0.5)
table(ReviewSample.train$polarity,pred.nnet, dnn=c("Truth","Predicted"))

#MCR
(nnet.insampleMCR<-mean(ReviewSample.train$polarity!=pred.nnet))

#ROC
library(ROCR)
pred <- prediction(prob.nnet,ReviewSample.train$polarity)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#AUC
unlist(slot(performance(pred, "auc"), "y.values"))

#Out-of-Sample
prob.nnet= predict(reviews.nnet,ReviewSample.test)
pred.nnet = as.numeric(prob.nnet > 0.5)
table(ReviewSample.test$polarity,pred.nnet, dnn=c("Truth","Predicted"))

#MCR
(nnet.outsampleMCR<-mean(ReviewSample.test$polarity!=pred.nnet))

#ROC
library(ROCR)
pred <- prediction(prob.nnet,ReviewSample.test$polarity)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#AUC
unlist(slot(performance(pred, "auc"), "y.values"))

#####################################################################
