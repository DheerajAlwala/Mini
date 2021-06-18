
# Importing libraries
library(caret) #Classification And Regression Training package
library(randomForest) #Used for classification and regression.
library(e1071) #Used for SVM 
library(kernlab) #Kernel-based machine learning methods
library(doMC) #Provides a parallel backend for the %dopar% function 
library(foreach) #looping
library(RColorBrewer) #tools to manage colors
library(parallel) #for parallel execution 
library(LiblineaR) #predictive linear models for classification and regression(regularized logistic regression)
library(verification) #Utilities for verifying predictions & forecasts
library(cvAUC) #for estimating cross-validated Area Under the ROC Curve (AUC)
library(gplots) #Utilities for Drawing Plots

# require(caret)
# require(parallel)
# require(foreach)
 
## Reading Data

# Reading the train data:
trFC <- read.csv('C:\\Users\\Surya\\Desktop\\Datasets\\DD\\train_FNC.csv')
trSBM <- read.csv('C:\\Users\\Surya\\Desktop\\Datasets\\DD\\train_SBM.csv')

# Merging trFC and trSBM by 'Id':
tr <- merge(trFC, trSBM, by='Id')

# Reading the test data:
tstFC <- read.csv('C:\\Users\\Surya\\Desktop\\Datasets\\DD\\test_FNC.csv')
tstSBM <- read.csv('C:\\Users\\Surya\\Desktop\\Datasets\\DD\\test_SBM.csv')

# Merging tstFC and tstSBM by 'Id':
tst <- merge(tstFC, tstSBM, by='Id')

# Reading labels of train data:
y <- read.csv('C:\\Users\\Surya\\Desktop\\Datasets\\DD\\train_labels.csv')
true_labels<- read.csv("C:\\Users\\Surya\\Desktop\\Datasets\\DD\\test_label.csv")

## I.Logistic regression

# Setting seed that R used to generate that sequence.(e.g.rnorm)
set.seed(3433)
q<-merge(tr,y,by="Id")

# Spliting the data with createDataPartition
inTrain = createDataPartition(q$Class, p = 3/4)[[1]]
training = q[ inTrain,]
testing = q[-inTrain,]

training$Class<-as.numeric(training$Class)
testing$Class<-as.numeric(testing$Class)

# Converting into appropriate format
j<-as.matrix(training[1:65,2:411])
w<-as.matrix(testing[,2:411])

# Training logistic regression model using all features
# Type-6-> L1-regularized logistic regression,
libfit<-LiblineaR(j, training[,412], type=6, cost=0.16,bias = TRUE, wi = NULL, verbose = TRUE)

#predtest<-predict(libfit,w,proba=TRUE)
#confusionMatrix(predtest$predictions, testing$Class)

# Making predictions
Test<-as.matrix(tst[,2:411])

predtest<-predict(libfit,Test,proba=TRUE)
prob<-predtest[[2]][,1]

# Saving predictions
predictions<-read.csv("C:\\Users\\Surya\\Desktop\\Datasets\\DD\\example.csv")
predictions[,2]<-prob

write.csv(predictions,"C:\\Users\\Surya\\Desktop\\Datasets\\DD\\lr_label.csv",row.names=FALSE)

# Replacing probabilities with 0 or 1
for (i in 1:nrow(predictions)){
  if (predictions$Probability[i] <= 0.6)
    predictions$Probability[i] <- 0
  else
    predictions$Probability
  [i]<-1
}

# Evaluation metrics
auc <- AUC(predictions$Probability,true_labels$Probability)# AUC score
roc.plot(predictions$Probability,true_labels$Probability)# ROC plot
mean(predictions$Probability == true_labels$Probability)# Accuracy
# Error rate

y_pred<-predictions$Probability
y_act<-true_labels$Probability

# Confusion matrix
xtable<-table(y_pred,y_act)
caret::confusionMatrix(xtable, mode="everything")



## II.RandomForest and RBF-SVM

# "Feature trimming"

# Registering 6 cores to speed up computations:
# detectCores()
registerDoMC(cores=12)

# Just converting a y-label vector into appropriate format:
y <- as.factor(paste('X.', y[,2], sep = ''))
# paste('X.',y[,2],sep = '')

# Introducing a random vector into the feature set):
all <- cbind(tr, rnorm(1:dim(tr)[1]))
colnames(all)[412] <- 'rand'

# training Random Forest with this (full) feature set:
rf.mod <- foreach(ntree=rep(2500, 6), .combine=combine, .multicombine=TRUE,
                  .packages='randomForest') %dopar% {
                    randomForest(all[,2:412], y, ntree=ntree)
                  }

# looking at the importance of each feature:
imp <- as.data.frame(rf.mod$importance[order(rf.mod$importance),])

# Everything below importance of my "dummy" feature (random vector) can likely be ignored
imp <- subset(imp, imp>imp['rand',])

# Saving the data in one rda-file for further analyses: 
save('all', 'y', 'tst', 'imp',  file = 'C:\\Users\\Surya\\Desktop\\Datasets\\DD\\AllData.rda')


# reducing feature set:
dat <- all[,rownames(imp)]


# Training the final model:

# Estimating "sigma" (inverse width parameter for the RBF-SVM)
sigDist <- sigest(y ~ as.matrix(dat), data=dat, frac = 1)

# Creating a tune grid for further C-parameter selection):
svmTuneGrid <- data.frame(.sigma = sigDist[1], .C = 2^(-20:100))

# And... training the final RBF-SVM model with leave-one-out cross-validation:
svmFit <- train(dat,y,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneGrid = svmTuneGrid,
                trControl = trainControl(method = "cv", number = 86, classProbs =  TRUE))

# Making predictions
ttst <- tst[,rownames(imp)]
predTst <- predict(svmFit, ttst, type='prob')
predTst <- predTst[,2]

# Saving predictions
predictions<-read.csv("C:\\Users\\Surya\\Desktop\\Datasets\\DD\\example.csv")
predictions[,2]<-predTst
write.csv(predictions,"C:\\Users\\Surya\\Desktop\\Datasets\\DD\\SVMFIT_label.csv",row.names=FALSE)

# Replacing probabilities with 0 or 1
for (i in 1:nrow(predictions)){
  if (predictions$Probability[i] <= 0.6)
    predictions$Probability[i] <- 0
  else
    predictions$Probability[i]<-1
}

y_predS<-predictions$Probability
y_actS<-true_labels$Probability

# Evaluation metrics
auc <- AUC(y_predS,y_actS)# AUC score
roc.plot(y_predS,y_actS)# ROC plot
mean(y_predS == y_actS)# accuracy

# Confusion matrix
xtable<-table(y_predS,y_actS)
caret::confusionMatrix(xtable, mode="everything")

# Plots
for (i in 1:nrow(imp)){
  imp$`rf.mod$importance[order(rf.mod$importance), ]`<-round(imp$`rf.mod$importance[order(rf.mod$importance), ]`,digits = 1)
}
hm <- data.matrix(imp)

# heatmap.2(cbind(hm, hm),main = "Importance", trace="n", Colv = NA,Scale="column",dendrogram = "row", labCol = "", labRow = hm, cexRow = 0.75,key = FALSE)

# Histogram
hist(hm,main = "Histogram of Importance",xlab="Feature Importance Score",ylab = "Frequency")

# Density Plot
plot(density(hm),cex=0.1,main="Density Plot of Importance",bty="n",ylab="no. of features",xlab="Importance")
polygon(density(hm),col="lightblue")
abline(v=mean(hm),col="red")#Mean
abline(v=mean(hm)+sd(hm),col="green")#Upper SD
abline(v=mean(hm)-sd(hm),col="blue")#Lower SD
legend("topright",legend=c("Mean","Lower SD","Upper SD"),col = c("red","blue","green"),lwd =6)
