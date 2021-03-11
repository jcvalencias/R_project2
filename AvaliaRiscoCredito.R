# Building a predictive model of credit risk

# Setting working directory
setwd("C:/Users/jcval/Documents/Projectos DSA/R_Project2")
getwd()

# Loading dataset into a dataframe
credit.df <- read.csv("credit_dataset.csv", header = TRUE, sep = ",")
head(credit.df)

## Function to change type of factor variables
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

## Function to set Normalization of variables
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizing features
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
credit.df <- scale.features(credit.df, numeric.vars)

# Factor variables
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

credit.df <- to.factors(df = credit.df, variables = categorical.vars)

# Splitting data into train and test - 60:40 ratio
indexes <- sample(1:nrow(credit.df), size = 0.6 * nrow(credit.df))
train.data <- credit.df[indexes,]
test.data <- credit.df[-indexes,]

# Feature Selection
library(caret) 
library(randomForest) 

# Function that performs the feature selection
run.feature.selection <- function(num.iters=20, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
                     sizes = variable.sizes, 
                     rfeControl = control)
  return(results.rfe)
}

# Executing the function
rfe.results <- run.feature.selection(feature.vars = train.data[,-1], 
                                     class.var = train.data[,1])


# Obtaining results
rfe.results
varImp((rfe.results))


# Creation and evaluation of model
library(caret) 
library(ROCR) 

# Library of graphic utilities
source("plot_utils.R") 

## separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# Building model of logistic regression
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
lr.model <- glm(formula = formula.init, data = train.data, family = "binomial")

# Visualizing model
summary(lr.model)

# Testing model with test data
lr.predictions <- predict(lr.model, test.data, type="response")
lr.predictions <- round(lr.predictions)

# Evaluating model
confusionMatrix(table(data = lr.predictions, reference = test.class.var), positive = '1')

## Feature selection
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = train.data, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)
plot(importance)


# Building model with selected features
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months"
formula.new <- as.formula(formula.new)
lr.model.new <- glm(formula = formula.new, data = train.data, family = "binomial")

# Visualizing model
summary(lr.model.new)

# Testing model with test data
lr.predictions.new <- predict(lr.model.new, test.data, type = "response") 
lr.predictions.new <- round(lr.predictions.new)

# Evaluating model
confusionMatrix(table(data = lr.predictions.new, reference = test.class.var), positive = '1')

# Evaluating model performance

# creating ROC curve
lr.model.best <- lr.model
lr.prediction.values <- predict(lr.model.best, test.feature.vars, type = "response")
predictions <- prediction(lr.prediction.values, test.class.var)
par(mfrow = c(1,2))
plot.roc.curve(predictions, title.text = "ROC Curve")
plot.pr.curve(predictions, title.text = "Precition/Recall Curve")

