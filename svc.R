library(dplyr)
library(tidyr)
library(quanteda)
library(quanteda.textmodels)
library(e1071)


svm_classifer_train <- function(docs_df, dtm, nd = 10) {
  
  # SVD / LSA
  lsa <- textmodel_lsa(dtm, nd = nd)
  d_vec <- as.data.frame(lsa$docs)
  d_vec[["id"]] <- row.names(lsa$docs)
  d_svc <- left_join(docs_df, d_vec, by = c("id" = "id")) %>%
    select(topic, starts_with("V")) %>%
    mutate(topic = as.factor(topic))  
  
  # Train-test split (80% train, 20% test)
  set.seed(11)
  train_idx <- sample(1:nrow(d_svc), 
                      size = as.integer(nrow(d_svc)*0.2))
  d_train <- d_svc[train_idx, ]
  d_test <- d_svc[-train_idx, ]
  
  # Fit SVC: Hyper-parameter tuning
  tune.model <- tune(svm,
                     topic ~ ., 
                     data = d_train,
                     kernel = "linear", # RBF kernel function
                     range = list(cost = seq(0.01, 5, by = 0.05))
  )
  
  # Evaluate Model
  train.pred <- predict(tune.model$best.model, d_train)
  test.pred <- predict(tune.model$best.model, d_test)
  confus.matrix <- table(real = d_train$topic, predict = train.pred)
  train_acc <- sum(diag(confus.matrix)) / sum(confus.matrix)
  confus.matrix <- table(real = d_test$topic, predict = test.pred)
  test_acc <- sum(diag(confus.matrix)) / sum(confus.matrix)
  cat('======== Model Performance: nd =', nd, '========\n')
  cat('Train accuracy:', round(train_acc, 4))
  cat('\tTest accuracy:', round(test_acc, 4))
  cat('\n\n')
  
  return(list(model = tune.model$best.model,
              train_acc = train_acc,
              test_acc = test_acc,
              nd = nd)
         )
}

