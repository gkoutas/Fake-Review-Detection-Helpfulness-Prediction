# STRUCTURE OF THIS PIPELINE:
# 1. Data Cleaning & Preprocessing
# 2. Feature Engineering (Sentiment, Readability, Subjectivity)
# 3. Text Mining (TF-IDF)
# 4. Stratified Data Splitting
# 5. Model Training & Tuning (LASSO, Random Forest, XGBoost)
# 6. Model Evaluation & Comparison

# -------- Load libraries --------
library(dplyr)
library(lubridate)
library(lexicon)
library(textclean)
library(sentimentr)
library(ggplot2)
library(caret)
library(tm)
library(tidytext)
library(Matrix)
library(glmnet)
library(ranger)
library(xgboost)
library(pROC)

# -------- Load data --------
reviews_df <- read.csv("data/reviews_dataset.csv")

### ==========================================================================================
### DATA CLEANING 
### ==========================================================================================

# Remove duplicates and short reviews
reviews_df <- reviews_df %>% 
  distinct(reviewID, .keep_all = TRUE) %>% 
  filter(reviewContent != "" & nchar(reviewContent) > 20)

# Convert dates
reviews_df <- reviews_df %>% 
  mutate(
    date = as.Date(date, format = "%m/%d/%Y"),
    yelpJoinDate = as.Date(yelpJoinDate, format = "%m/%d/%Y")
  )

### ==========================================================================================
### FAKE VARIABLES
### ==========================================================================================

# Create fake binary
reviews_df <- reviews_df %>% 
  mutate(fake_binary = ifelse(flagged == "Y", 1, 0))

### ==========================================================================================
### CREATE HELPFULNESS SCORE
### ==========================================================================================

# 1. Explore all engagement metrics
engagement_metrics <- c("reviewUsefulCount", "coolCount", "funnyCount", "complimentCount")
cat("=== DISTRIBUTION OF ENGAGEMENT METRICS ===\n")
print(summary(reviews_df[, engagement_metrics]))

# 2. Composite helpfulness score (continuous)
reviews_df$helpfulness_score <- (
  reviews_df$reviewUsefulCount * 0.5 +
    reviews_df$coolCount        * 0.3 +
    reviews_df$complimentCount  * 0.2 +
    reviews_df$funnyCount       * 0.1
)
cat("=== NEW HELPFULNESS SCORE DISTRIBUTION ===\n")
print(summary(reviews_df$helpfulness_score))
hist(reviews_df$helpfulness_score, main = "Composite Helpfulness Score", breaks = 50)

# 3. Binary helpfulness (top 30%)
helpfulness_threshold <- quantile(reviews_df$helpfulness_score, 0.7, na.rm = TRUE)
cat("Helpfulness threshold (70th percentile):", helpfulness_threshold, "\n")
reviews_df$helpful_binary <- ifelse(reviews_df$helpfulness_score > helpfulness_threshold, 1, 0)

# 4–7. Checks & visualization
cat("=== NEW HELPFULNESS FOR FAKE vs REAL REVIEWS ===\n")
helpful_table <- table(reviews_df$helpful_binary, reviews_df$fake_binary)
print(helpful_table)
print(prop.table(helpful_table, margin = 2))

cat("Enhanced helpfulness distribution:\n")
cat("Fake reviews labeled as helpful:", sum(reviews_df$helpful_binary[reviews_df$fake_binary == 1]), "\n")
cat("Real reviews labeled as helpful:", sum(reviews_df$helpful_binary[reviews_df$fake_binary == 0]), "\n")
cat("Percentage of fake reviews that are helpful:", 
    mean(reviews_df$helpful_binary[reviews_df$fake_binary == 1]) * 100, "%\n")

engagement_plot_data <- data.frame(
  Metric = rep(engagement_metrics, 2),
  Value  = c(sapply(engagement_metrics, function(m) mean(reviews_df[[m]][reviews_df$fake_binary == 1], na.rm=TRUE)),
             sapply(engagement_metrics, function(m) mean(reviews_df[[m]][reviews_df$fake_binary == 0], na.rm=TRUE))),
  Type   = rep(c("Fake", "Real"), each = length(engagement_metrics))
)

ggplot(engagement_plot_data, aes(x = Metric, y = Value, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Average Engagement Metrics: Fake vs Real Reviews",
       x = "Engagement Metric", y = "Average Count") +
  theme_minimal() +
  scale_fill_manual(values = c("Fake" = "red", "Real" = "blue"))

### ==========================================================================================
### POLARITY VARIABLE
### ==========================================================================================
reviews_df <- reviews_df %>%
  mutate(
    review_sentences = get_sentences(reviewContent),
    polarity = sentiment_by(review_sentences)$ave_sentiment
  )

### ==========================================================================================
### SUBJECTIVITY VARIABLE
### ==========================================================================================
data(hash_sentiment_jockers_rinker)
opinion_words <- hash_sentiment_jockers_rinker$x

reviews_df <- reviews_df %>%
  mutate(
    clean_text = tolower(gsub("[^A-Za-z ]", "", reviewContent)),
    opinion_word_count = sapply(strsplit(clean_text, " "), function(words) sum(words %in% opinion_words)),
    subjectivity = opinion_word_count / (sapply(strsplit(clean_text, " "), length) + 1e-6)
  ) %>%
  select(-clean_text)

### ==========================================================================================
### READABILITY VARIABLE
### ==========================================================================================
calculate_readability <- function(text) {
  if (nchar(text) < 10) return(NA_real_)
  chars <- nchar(gsub("[^A-Za-z0-9]", "", text))
  words <- strsplit(tolower(gsub("[^A-Za-z ]", "", text)), " ")[[1]]
  words <- words[words != ""]
  word_count <- length(words)
  sentences <- max(1, lengths(gregexpr("[.!?]", text)))
  ari <- 4.71 * (chars / word_count) + 0.5 * (word_count / sentences) - 21.43
  round(pmax(pmin(ari, 14), 1), 1)
}
reviews_df$readability <- sapply(reviews_df$reviewContent, calculate_readability)

### ==========================================================================================
### DEPTH VARIABLE
### ==========================================================================================
reviews_df <- reviews_df %>%
  mutate(depth_wordcount = lengths(strsplit(reviewContent, "\\s+"))) %>%
  filter(!is.na(readability))

### ==========================================================================================
### TRANSFORMATIONS OF DEPTH AND READABILITY BEFORE MODELING
### (Supervisor: min–max for readability, log+standardize for depth)
### ==========================================================================================

# Keep originals; create transformed versions used by models
reviews_df <- reviews_df %>%
  mutate(
    # log-transform word counts, then standardize (z-score)
    depth_log = as.numeric(scale(log1p(depth_wordcount))), 
    
    # min–max normalization for readability (already on 1–14 scale but we rescale to [0,1])
    readability_norm = (readability - min(readability, na.rm = TRUE)) /
      (max(readability, na.rm = TRUE) - min(readability, na.rm = TRUE))
  )


### ==========================================================================================
### TF-IDF FEATURE ENGINEERING
### ==========================================================================================
corpus <- VCorpus(VectorSource(reviews_df$reviewContent)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
term_sums <- colSums(as.matrix(dtm))
top_terms <- names(sort(term_sums, decreasing = TRUE)[1:1000])
tfidf_reduced <- as.matrix(dtm[, top_terms])
colnames(tfidf_reduced) <- paste0("tfidf_", make.names(colnames(tfidf_reduced)))

if (nrow(reviews_df) != nrow(tfidf_reduced)) stop("Row mismatch!")
reviews_df <- cbind(reviews_df, tfidf_reduced)

### ==========================================================================================
### FINAL DATA PREPROCESSING PIPELINE
### ==========================================================================================
cat("=== CORRELATION: ENGAGEMENT vs REVIEW CHARACTERISTICS ===\n")
correlation_vars <- c("helpfulness_score", "polarity", "subjectivity",
                      "readability_norm", "depth_log", "rating")
cor_matrix <- cor(reviews_df[, correlation_vars], use = "complete.obs")
print(cor_matrix)

content_features <- c("polarity", "subjectivity", "readability_norm", "depth_log", "rating")
tfidf_cols <- grep("^tfidf_", names(reviews_df), value = TRUE)
sparsity_check <- sapply(reviews_df[, tfidf_cols], function(x) sum(x == 0) / length(x))
keep_tfidf <- names(sparsity_check[sparsity_check <= 0.95])

thesis_features <- c("fake_binary", "helpful_binary", content_features, keep_tfidf)
reviews_thesis <- reviews_df[, thesis_features]
reviews_thesis$strata_label <- paste0("F", reviews_thesis$fake_binary, "_H", reviews_thesis$helpful_binary)

# Split data once (stratified on combined label)
set.seed(123)
train_idx <- createDataPartition(reviews_thesis$strata_label, p = 0.8, list = FALSE)
train_data <- reviews_thesis[train_idx, ]
test_data  <- reviews_thesis[-train_idx, ]

X_train <- train_data[, setdiff(names(train_data), c("fake_binary", "helpful_binary", "strata_label"))]
X_test  <- test_data[, setdiff(names(test_data), c("fake_binary", "helpful_binary", "strata_label"))]
y_train_fake <- train_data$fake_binary
y_test_fake  <- test_data$fake_binary
y_train_helpful <- train_data$helpful_binary
y_test_helpful  <- test_data$helpful_binary

cat("Train helpful split:\n"); print(prop.table(table(y_train_helpful)))
cat("Test helpful split:\n");  print(prop.table(table(y_test_helpful)))
cat("Train fake split:\n");    print(prop.table(table(y_train_fake)))
cat("Test fake split:\n");     print(prop.table(table(y_test_fake)))

# -------- Shared CV folds (for LASSO & XGBoost). RF uses OOB. --------
set.seed(42)
K <- 5
folds_fake <- caret::createFolds(y = y_train_fake, k = K, list = TRUE, returnTrain = FALSE)
foldid_fake <- integer(length(y_train_fake)); for (i in seq_along(folds_fake)) foldid_fake[folds_fake[[i]]] <- i
folds_helpful <- caret::createFolds(y = y_train_helpful, k = K, list = TRUE, returnTrain = FALSE)
foldid_helpful <- integer(length(y_train_helpful)); for (i in seq_along(folds_helpful)) foldid_helpful[folds_helpful[[i]]] <- i

### ==========================================================================================
### LASSO REGRESSION (alpha = 1, threshold = 0.5, shared folds)
### ==========================================================================================

set.seed(123)
# ---- LASSO Fake ----
lasso_cv_fake <- cv.glmnet(
  x = model.matrix(~ ., X_train)[, -1],
  y = y_train_fake,
  alpha = 1,
  family = "binomial",
  type.measure = "auc",
  foldid = foldid_fake
)
best_lambda_fake <- lasso_cv_fake$lambda.min
pred_prob_fake <- predict(lasso_cv_fake, newx = model.matrix(~ ., X_test)[, -1],
                          s = best_lambda_fake, type = "response")
pred_class_fake <- ifelse(pred_prob_fake > 0.5, 1, 0)
cm_fake <- caret::confusionMatrix(factor(pred_class_fake), factor(y_test_fake), positive = "1")
precision_lasso_fake <- cm_fake$byClass["Precision"]
recall_lasso_fake    <- cm_fake$byClass["Recall"]
f1_lasso_fake        <- cm_fake$byClass["F1"]
accuracy_lasso_fake  <- cm_fake$overall["Accuracy"]
auc_lasso_fake       <- pROC::auc(pROC::roc(y_test_fake, as.vector(pred_prob_fake)))
cat("\n=== FINAL LASSO – Fake ===\n"); print(cm_fake$table)
cat(sprintf("Accuracy %.3f  Precision %.3f  Recall %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_lasso_fake, precision_lasso_fake, recall_lasso_fake, f1_lasso_fake, auc_lasso_fake))

# Feature importance (non-zero coefficients)
coef_df_fake <- as.data.frame(as.matrix(coef(lasso_cv_fake, s = best_lambda_fake)))
coef_df_fake$feature <- rownames(coef_df_fake)
colnames(coef_df_fake)[1] <- "coefficient"
coef_df_fake <- coef_df_fake %>%
  filter(coefficient != 0 & feature != "(Intercept)") %>%
  arrange(desc(abs(coefficient)))
cat("\nTop 10 LASSO Features – Fake:\n"); print(head(coef_df_fake, 10))
ggplot(head(coef_df_fake, 10), aes(x = reorder(feature, abs(coefficient)), y = abs(coefficient))) +
  geom_bar(stat = "identity", fill = "darkred") + coord_flip() +
  labs(title = "Top 10 Features – LASSO (Fake)", x = "Feature", y = "Coefficient |value|") + theme_minimal()
plot(lasso_cv_fake, xvar = "lambda", label = TRUE); title("LASSO Coefficient Paths – Fake")

# ---- LASSO Helpfulness ----
lasso_cv_helpful <- cv.glmnet(
  x = model.matrix(~ ., X_train)[, -1],
  y = y_train_helpful,
  alpha = 1,
  family = "binomial",
  type.measure = "auc",
  foldid = foldid_helpful
)
best_lambda_helpful <- lasso_cv_helpful$lambda.min
pred_prob_helpful <- predict(lasso_cv_helpful, newx = model.matrix(~ ., X_test)[, -1],
                             s = best_lambda_helpful, type = "response")
pred_class_helpful <- ifelse(pred_prob_helpful > 0.5, 1, 0)
cm_helpful <- caret::confusionMatrix(factor(pred_class_helpful), factor(y_test_helpful), positive = "1")
precision_lasso_helpful <- cm_helpful$byClass["Precision"]
recall_lasso_helpful    <- cm_helpful$byClass["Recall"]
f1_lasso_helpful        <- cm_helpful$byClass["F1"]
accuracy_lasso_helpful  <- cm_helpful$overall["Accuracy"]
auc_lasso_helpful       <- pROC::auc(pROC::roc(y_test_helpful, as.vector(pred_prob_helpful)))
cat("\n=== FINAL LASSO – Helpfulness ===\n"); print(cm_helpful$table)
cat(sprintf("Accuracy %.3f  Precision %.3f  Recall %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_lasso_helpful, precision_lasso_helpful, recall_lasso_helpful, f1_lasso_helpful, auc_lasso_helpful))

coef_df_helpful <- as.data.frame(as.matrix(coef(lasso_cv_helpful, s = best_lambda_helpful)))
colnames(coef_df_helpful)[1] <- "coefficient"
coef_df_helpful$feature <- rownames(coef_df_helpful)
coef_df_helpful <- coef_df_helpful %>%
  filter(coefficient != 0 & feature != "(Intercept)") %>%
  arrange(desc(abs(coefficient)))
cat("\nTop 10 LASSO Features – Helpfulness:\n"); print(head(coef_df_helpful, 10))
ggplot(head(coef_df_helpful, 10), aes(x = reorder(feature, abs(coefficient)), y = abs(coefficient))) +
  geom_bar(stat = "identity", fill = "darkblue") + coord_flip() +
  labs(title = "Top 10 Features – LASSO (Helpfulness)", x = "Feature", y = "Coefficient |value|") + theme_minimal()
plot(lasso_cv_helpful, xvar = "lambda", label = TRUE); title("LASSO Coefficient Paths – Helpfulness")


### COEFFICIENTS OF FITTED LASSO MODELS

# Fake Detection
coef_df_fake <- as.data.frame(as.matrix(coef(lasso_cv_fake, s = best_lambda_fake)))
coef_df_fake$feature <- rownames(coef_df_fake)
colnames(coef_df_fake)[1] <- "coefficient"

coef_df_fake <- coef_df_fake %>%
  filter(coefficient != 0 & feature != "(Intercept)") %>%
  arrange(desc(abs(coefficient)))

# Plot actual coefficients (signed)
ggplot(head(coef_df_fake, 10), aes(x = reorder(feature, coefficient), y = coefficient)) +
  geom_bar(stat = "identity", fill = "darkred") + coord_flip() +
  labs(title = "Top 10 LASSO Coefficients – Fake Detection",
       x = "Feature", y = "Coefficient") +
  theme_minimal()


### ==========================================================================================
### RANDOM FOREST — tune ONLY mtry (OOB), 0.5 threshold
### ==========================================================================================

set.seed(123)
p <- ncol(X_train)
mtry_start <- floor(sqrt(p))
mtry_values <- sort(unique(pmax(1, round(mtry_start * c(0.25, 0.5, 0.75, 1, 1.25, 1.5, 2)))))

# ---- RF Fake ----
oob_errors_fake <- numeric(length(mtry_values))
rf_models_fake  <- vector("list", length(mtry_values))
for (i in seq_along(mtry_values)) {
  rf_model <- ranger(
    y_train_fake ~ .,
    data = cbind(y_train_fake = factor(y_train_fake), X_train),
    num.trees = 1000,
    mtry = mtry_values[i],
    probability = TRUE,
    importance = "permutation",
    seed = 123,
    oob.error = TRUE
  )
  oob_errors_fake[i] <- rf_model$prediction.error
  rf_models_fake[[i]] <- rf_model
}
oob_df_fake <- data.frame(mtry = mtry_values, OOB_Error = oob_errors_fake)
ggplot(oob_df_fake, aes(x = mtry, y = OOB_Error)) + geom_line() + geom_point(size = 3) +
  labs(title = "RF OOB Error vs mtry (Fake Detection)", x = "mtry", y = "OOB Error") + theme_minimal()
best_idx_fake <- which.min(oob_errors_fake)
rf_best_fake  <- rf_models_fake[[best_idx_fake]]
cat("Best mtry for Fake:", mtry_values[best_idx_fake], "\n")

pr_rf_fake <- predict(rf_best_fake, data = X_test)$predictions[, "1"]
pred_rf_fake <- ifelse(pr_rf_fake > 0.5, 1, 0)
cm_rf_fake <- caret::confusionMatrix(factor(pred_rf_fake), factor(y_test_fake), positive = "1")
precision_rf_fake <- as.numeric(cm_rf_fake$byClass["Precision"])
recall_rf_fake    <- as.numeric(cm_rf_fake$byClass["Recall"])
f1_rf_fake        <- ifelse(precision_rf_fake + recall_rf_fake > 0, 2*precision_rf_fake*recall_rf_fake/(precision_rf_fake+recall_rf_fake), NA)
accuracy_rf_fake  <- as.numeric(cm_rf_fake$overall["Accuracy"])
auc_rf_fake       <- pROC::auc(pROC::roc(y_test_fake, as.numeric(pr_rf_fake)))
cat("\n=== FINAL RF – Fake (0.5 thr) ===\n"); print(cm_rf_fake$table)
cat(sprintf("Acc %.3f  Prec %.3f  Rec %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_rf_fake, precision_rf_fake, recall_rf_fake, f1_rf_fake, auc_rf_fake))

importance_fake <- sort(ranger::importance(rf_best_fake), decreasing = TRUE)
cat("\nTop 15 RF Features – Fake:\n"); print(head(importance_fake, 15))
importance_df_fake <- data.frame(Feature = names(importance_fake), Importance = as.numeric(importance_fake))
ggplot(importance_df_fake[1:15, ], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "darkred") + coord_flip() +
  labs(title = "Top 15 Important Features – RF (Fake)", x = "Feature", y = "Importance") + theme_minimal()

# ---- RF Helpfulness ----
oob_errors_helpful <- numeric(length(mtry_values))
rf_models_helpful  <- vector("list", length(mtry_values))
for (i in seq_along(mtry_values)) {
  rf_model <- ranger(
    y_train_helpful ~ .,
    data = cbind(y_train_helpful = factor(y_train_helpful), X_train),
    num.trees = 1000,
    mtry = mtry_values[i],
    probability = TRUE,
    importance = "permutation",
    seed = 123,
    oob.error = TRUE
  )
  oob_errors_helpful[i] <- rf_model$prediction.error
  rf_models_helpful[[i]] <- rf_model
}
oob_df_helpful <- data.frame(mtry = mtry_values, OOB_Error = oob_errors_helpful)
ggplot(oob_df_helpful, aes(x = mtry, y = OOB_Error)) + geom_line() + geom_point(size = 3) +
  labs(title = "RF OOB Error vs mtry (Helpfulness)", x = "mtry", y = "OOB Error") + theme_minimal()
best_idx_helpful <- which.min(oob_errors_helpful)
rf_best_helpful  <- rf_models_helpful[[best_idx_helpful]]
cat("Best mtry for Helpfulness:", mtry_values[best_idx_helpful], "\n")

pr_rf_helpful <- predict(rf_best_helpful, data = X_test)$predictions[, "1"]
pred_rf_helpful <- ifelse(pr_rf_helpful > 0.5, 1, 0)
cm_rf_helpful <- caret::confusionMatrix(factor(pred_rf_helpful), factor(y_test_helpful), positive = "1")
precision_rf_helpful <- as.numeric(cm_rf_helpful$byClass["Precision"])
recall_rf_helpful    <- as.numeric(cm_rf_helpful$byClass["Recall"])
f1_rf_helpful        <- ifelse(precision_rf_helpful + recall_rf_helpful > 0, 2*precision_rf_helpful*recall_rf_helpful/(precision_rf_helpful+recall_rf_helpful), NA)
accuracy_rf_helpful  <- as.numeric(cm_rf_helpful$overall["Accuracy"])
auc_rf_helpful       <- pROC::auc(pROC::roc(y_test_helpful, as.numeric(pr_rf_helpful)))
cat("\n=== FINAL RF – Helpfulness (0.5 thr) ===\n"); print(cm_rf_helpful$table)
cat(sprintf("Acc %.3f  Prec %.3f  Rec %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_rf_helpful, precision_rf_helpful, recall_rf_helpful, f1_rf_helpful, auc_rf_helpful))

importance_helpful <- sort(ranger::importance(rf_best_helpful), decreasing = TRUE)
cat("\nTop 15 RF Features – Helpfulness:\n"); print(head(importance_helpful, 15))
importance_df_helpful <- data.frame(Feature = names(importance_helpful), Importance = as.numeric(importance_helpful))
ggplot(importance_df_helpful[1:15, ], aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "darkblue") + coord_flip() +
  labs(title = "Top 15 Important Features – RF (Helpfulness)", x = "Feature", y = "Importance") + theme_minimal()

### ==========================================================================================
### BALANCED RF — reuse best mtry, add class.weights, 0.5 threshold
### ==========================================================================================
set.seed(123)
# ---- Balanced RF Fake ----
tab_fake <- table(factor(y_train_fake, levels = c(0,1)))
w0 <- sum(tab_fake) / (2 * as.numeric(tab_fake["0"]))
w1 <- sum(tab_fake) / (2 * as.numeric(tab_fake["1"]))
class_w_fake <- c("0" = w0, "1" = w1)

rf_bal_fake <- ranger(
  y_train_fake ~ .,
  data = cbind(y_train_fake = factor(y_train_fake), X_train),
  num.trees = 1000,
  mtry = mtry_values[best_idx_fake],
  probability = TRUE,
  importance = "permutation",
  seed = 123,
  oob.error = TRUE,
  class.weights = class_w_fake
)
pr_rf_bal_fake <- predict(rf_bal_fake, data = X_test)$predictions[, "1"]
pred_rf_bal_fake <- ifelse(pr_rf_bal_fake > 0.5, 1, 0)
cm_rf_bal_fake <- caret::confusionMatrix(factor(pred_rf_bal_fake), factor(y_test_fake), positive = "1")
precision_rf_bal_fake <- as.numeric(cm_rf_bal_fake$byClass["Precision"])
recall_rf_bal_fake    <- as.numeric(cm_rf_bal_fake$byClass["Recall"])
f1_rf_bal_fake        <- ifelse(precision_rf_bal_fake + recall_rf_bal_fake > 0, 2*precision_rf_bal_fake*recall_rf_bal_fake/(precision_rf_bal_fake+recall_rf_bal_fake), NA)
accuracy_rf_bal_fake  <- as.numeric(cm_rf_bal_fake$overall["Accuracy"])
auc_rf_bal_fake       <- pROC::auc(pROC::roc(y_test_fake, as.numeric(pr_rf_bal_fake)))
cat("\n=== BALANCED RF – Fake (0.5 thr) ===\n"); print(cm_rf_bal_fake$table)
cat(sprintf("Acc %.3f  Prec %.3f  Rec %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_rf_bal_fake, precision_rf_bal_fake, recall_rf_bal_fake, f1_rf_bal_fake, auc_rf_bal_fake))

# ---- Balanced RF Helpfulness ----
tab_help <- table(factor(y_train_helpful, levels = c(0,1)))
w0h <- sum(tab_help) / (2 * as.numeric(tab_help["0"]))
w1h <- sum(tab_help) / (2 * as.numeric(tab_help["1"]))
class_w_help <- c("0" = w0h, "1" = w1h)

rf_bal_helpful <- ranger(
  y_train_helpful ~ .,
  data = cbind(y_train_helpful = factor(y_train_helpful), X_train),
  num.trees = 1000,
  mtry = mtry_values[best_idx_helpful],
  probability = TRUE,
  importance = "permutation",
  seed = 123,
  oob.error = TRUE,
  class.weights = class_w_help
)
pr_rf_bal_help <- predict(rf_bal_helpful, data = X_test)$predictions[, "1"]
pred_rf_bal_help <- ifelse(pr_rf_bal_help > 0.5, 1, 0)
cm_rf_bal_help <- caret::confusionMatrix(factor(pred_rf_bal_help), factor(y_test_helpful), positive = "1")
precision_rf_bal_helpful <- as.numeric(cm_rf_bal_help$byClass["Precision"])
recall_rf_bal_helpful    <- as.numeric(cm_rf_bal_help$byClass["Recall"])
f1_rf_bal_helpful        <- ifelse(precision_rf_bal_helpful + recall_rf_bal_helpful > 0, 2*precision_rf_bal_helpful*recall_rf_bal_helpful/(precision_rf_bal_helpful+recall_rf_bal_helpful), NA)
accuracy_rf_bal_helpful  <- as.numeric(cm_rf_bal_help$overall["Accuracy"])
auc_rf_bal_helpful       <- pROC::auc(pROC::roc(y_test_helpful, as.numeric(pr_rf_bal_help)))
cat("\n=== BALANCED RF – Helpfulness (0.5 thr) ===\n"); print(cm_rf_bal_help$table)
cat(sprintf("Acc %.3f  Prec %.3f  Rec %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_rf_bal_helpful, precision_rf_bal_helpful, recall_rf_bal_helpful, f1_rf_bal_helpful, auc_rf_bal_helpful))

### ==========================================================================================
### XGBOOST — shared folds CV (AUC), threshold = 0.5
### ==========================================================================================
# Build fold index lists for xgb.cv
xgb_folds_fake <- lapply(folds_fake, function(idx) as.integer(idx))
xgb_folds_help <- lapply(folds_helpful, function(idx) as.integer(idx))

# ---- XGB Fake ----
dtrain_fake <- xgb.DMatrix(data = as.matrix(X_train), label = y_train_fake)
xgb_grid_fake <- expand.grid(
  eta = c(0.05, 0.10),
  max_depth = c(4, 6, 8),
  subsample = c(0.8, 1.0),
  colsample_bytree = c(0.8, 1.0),
  nrounds = c(300, 600)
)
xgb_res_fake <- list()
for (i in 1:nrow(xgb_grid_fake)) {
  g <- xgb_grid_fake[i, ]
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = g$eta,
    max_depth = g$max_depth,
    subsample = g$subsample,
    colsample_bytree = g$colsample_bytree
  )
  cv <- xgb.cv(
    params = params,
    data = dtrain_fake,
    nrounds = g$nrounds,
    folds = xgb_folds_fake,
    verbose = 0,
    early_stopping_rounds = 50
  )
  xgb_res_fake[[i]] <- data.frame(
    eta = g$eta, max_depth = g$max_depth, subsample = g$subsample,
    colsample_bytree = g$colsample_bytree, nrounds = g$nrounds,
    best_iter = cv$best_iteration,
    auc_mean = cv$evaluation_log$test_auc_mean[cv$best_iteration],
    auc_std  = cv$evaluation_log$test_auc_std[cv$best_iteration]
  )
}
xgb_res_df_fake <- dplyr::bind_rows(xgb_res_fake) %>% arrange(desc(auc_mean))
cat("\n=== XGB CV results (Fake) — top 10 ===\n"); print(head(xgb_res_df_fake, 10))

ggplot(xgb_res_df_fake, aes(x = factor(max_depth), y = factor(nrounds), fill = auc_mean)) +
  geom_tile() + 
  facet_wrap(~ eta + subsample + colsample_bytree, ncol = 2) +
  labs(title = "XGBoost CV AUC – Fake (shared folds)", x = "max_depth", y = "nrounds", fill = "AUC") +
  theme_minimal()

best_fake_xgb <- xgb_res_df_fake[1, ]
params_final_fake <- list(
  objective = "binary:logistic", eval_metric = "auc",
  eta = best_fake_xgb$eta, max_depth = best_fake_xgb$max_depth,
  subsample = best_fake_xgb$subsample, colsample_bytree = best_fake_xgb$colsample_bytree
)
final_xgb_fake <- xgb.train(params = params_final_fake, data = dtrain_fake,
                            nrounds = best_fake_xgb$best_iter, verbose = 0)
pr_xgb_fake <- predict(final_xgb_fake, xgb.DMatrix(as.matrix(X_test)))
pred_xgb_fake <- ifelse(pr_xgb_fake > 0.5, 1, 0)
cm_xgb_fake <- caret::confusionMatrix(factor(pred_xgb_fake), factor(y_test_fake), positive = "1")
precision_xgb_fake <- as.numeric(cm_xgb_fake$byClass["Precision"])
recall_xgb_fake    <- as.numeric(cm_xgb_fake$byClass["Recall"])
f1_xgb_fake        <- ifelse(precision_xgb_fake + recall_xgb_fake > 0, 2*precision_xgb_fake*recall_xgb_fake/(precision_xgb_fake+recall_xgb_fake), NA)
accuracy_xgb_fake  <- as.numeric(cm_xgb_fake$overall["Accuracy"])
auc_xgb_fake       <- pROC::auc(pROC::roc(y_test_fake, as.numeric(pr_xgb_fake)))
cat("\n=== FINAL XGB – Fake (0.5 thr) ===\n"); print(cm_xgb_fake$table)
cat(sprintf("Acc %.3f  Prec %.3f  Rec %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_xgb_fake, precision_xgb_fake, recall_xgb_fake, f1_xgb_fake, auc_xgb_fake))
imp_fake_xgb <- xgb.importance(model = final_xgb_fake, feature_names = colnames(X_train))
print(head(imp_fake_xgb, 15))
imp_fake_xgb_df <- imp_fake_xgb[1:15, c("Feature", "Gain")]
ggplot(imp_fake_xgb_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "darkred") + coord_flip() +
  labs(title = "Top 15 Important Features – XGBoost (Fake)", x = "Feature", y = "Gain") + theme_minimal()

# ---- XGB Helpfulness ----
dtrain_helpful <- xgb.DMatrix(data = as.matrix(X_train), label = y_train_helpful)
xgb_grid_help <- expand.grid(
  eta = c(0.05, 0.10),
  max_depth = c(4, 6, 8),
  subsample = c(0.8, 1.0),
  colsample_bytree = c(0.8, 1.0),
  nrounds = c(300, 600)
)
xgb_res_help <- list()
for (i in 1:nrow(xgb_grid_help)) {
  g <- xgb_grid_help[i, ]
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = g$eta,
    max_depth = g$max_depth,
    subsample = g$subsample,
    colsample_bytree = g$colsample_bytree
  )
  cv <- xgb.cv(
    params = params,
    data = dtrain_helpful,
    nrounds = g$nrounds,
    folds = xgb_folds_help,
    verbose = 0,
    early_stopping_rounds = 50
  )
  xgb_res_help[[i]] <- data.frame(
    eta = g$eta, max_depth = g$max_depth, subsample = g$subsample,
    colsample_bytree = g$colsample_bytree, nrounds = g$nrounds,
    best_iter = cv$best_iteration,
    auc_mean = cv$evaluation_log$test_auc_mean[cv$best_iteration],
    auc_std  = cv$evaluation_log$test_auc_std[cv$best_iteration]
  )
}
xgb_res_df_help <- dplyr::bind_rows(xgb_res_help) %>% arrange(desc(auc_mean))
cat("\n=== XGB CV results (Helpfulness) — top 10 ===\n"); print(head(xgb_res_df_help, 10))

ggplot(xgb_res_df_help, aes(x = factor(max_depth), y = factor(nrounds), fill = auc_mean)) +
  geom_tile() +
  facet_wrap(~ eta + subsample + colsample_bytree, ncol = 2) +
  labs(title = "XGBoost CV AUC – Helpfulness (shared folds)", x = "max_depth", y = "nrounds", fill = "AUC") +
  theme_minimal()

best_help_xgb <- xgb_res_df_help[1, ]
params_final_help <- list(
  objective = "binary:logistic", eval_metric = "auc",
  eta = best_help_xgb$eta, max_depth = best_help_xgb$max_depth,
  subsample = best_help_xgb$subsample, colsample_bytree = best_help_xgb$colsample_bytree
)
final_xgb_helpful <- xgb.train(params = params_final_help, data = dtrain_helpful,
                               nrounds = best_help_xgb$best_iter, verbose = 0)
pr_xgb_help <- predict(final_xgb_helpful, xgb.DMatrix(as.matrix(X_test)))
pred_xgb_help <- ifelse(pr_xgb_help > 0.5, 1, 0)
cm_xgb_help <- caret::confusionMatrix(factor(pred_xgb_help), factor(y_test_helpful), positive = "1")
precision_xgb_helpful <- as.numeric(cm_xgb_help$byClass["Precision"])
recall_xgb_helpful    <- as.numeric(cm_xgb_help$byClass["Recall"])
f1_xgb_helpful        <- ifelse(precision_xgb_helpful + recall_xgb_helpful > 0, 2*precision_xgb_helpful*recall_xgb_helpful/(precision_xgb_helpful+recall_xgb_helpful), NA)
accuracy_xgb_helpful  <- as.numeric(cm_xgb_help$overall["Accuracy"])
auc_xgb_helpful       <- pROC::auc(pROC::roc(y_test_helpful, as.numeric(pr_xgb_help)))
cat("\n=== FINAL XGB – Helpfulness (0.5 thr) ===\n"); print(cm_xgb_help$table)
cat(sprintf("Acc %.3f  Prec %.3f  Rec %.3f  F1 %.3f  AUC %.3f\n",
            accuracy_xgb_helpful, precision_xgb_helpful, recall_xgb_helpful, f1_xgb_helpful, auc_xgb_helpful))
imp_help_xgb <- xgb.importance(model = final_xgb_helpful, feature_names = colnames(X_train))
print(head(imp_help_xgb, 15))
imp_help_xgb_df <- imp_help_xgb[1:15, c("Feature", "Gain")]
ggplot(imp_help_xgb_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "darkblue") + coord_flip() +
  labs(title = "Top 15 Important Features – XGBoost (Helpfulness)", x = "Feature", y = "Gain") + theme_minimal()

### ==========================================================================================
### FINAL COMPARISON TABLES (all evaluated at threshold = 0.5)
### ==========================================================================================
# ----- Fake -----
comparison_fake <- data.frame(
  Model     = c("LASSO", "RF", "Balanced RF", "XGBoost"),
  Accuracy  = c(round(accuracy_lasso_fake, 3),   round(accuracy_rf_fake, 3),   round(accuracy_rf_bal_fake, 3),   round(accuracy_xgb_fake, 3)),
  Precision = c(round(precision_lasso_fake, 3),  round(precision_rf_fake, 3),  round(precision_rf_bal_fake, 3),  round(precision_xgb_fake, 3)),
  Recall    = c(round(recall_lasso_fake, 3),     round(recall_rf_fake, 3),     round(recall_rf_bal_fake, 3),     round(recall_xgb_fake, 3)),
  F1        = c(round(f1_lasso_fake, 3),         round(f1_rf_fake, 3),         round(f1_rf_bal_fake, 3),         round(f1_xgb_fake, 3)),
  AUC       = c(round(auc_lasso_fake, 3),        round(auc_rf_fake, 3),        round(auc_rf_bal_fake, 3),        round(auc_xgb_fake, 3))
)
cat("\n=== MODEL COMPARISON: Fake (0.5 thr) ===\n"); print(comparison_fake)

# ----- Helpfulness -----
comparison_helpful <- data.frame(
  Model     = c("LASSO", "RF", "Balanced RF", "XGBoost"),
  Accuracy  = c(round(accuracy_lasso_helpful, 3),   round(accuracy_rf_helpful, 3),   round(accuracy_rf_bal_helpful, 3),   round(accuracy_xgb_helpful, 3)),
  Precision = c(round(precision_lasso_helpful, 3),  round(precision_rf_helpful, 3),  round(precision_rf_bal_helpful, 3),  round(precision_xgb_helpful, 3)),
  Recall    = c(round(recall_lasso_helpful, 3),     round(recall_rf_helpful, 3),     round(recall_rf_bal_helpful, 3),     round(recall_xgb_helpful, 3)),
  F1        = c(round(f1_lasso_helpful, 3),         round(f1_rf_helpful, 3),         round(f1_rf_bal_helpful, 3),         round(f1_xgb_helpful, 3)),
  AUC       = c(round(auc_lasso_helpful, 3),        round(auc_rf_helpful, 3),        round(auc_rf_bal_helpful, 3),        round(auc_xgb_helpful, 3))
)
cat("\n=== MODEL COMPARISON: Helpfulness (0.5 thr) ===\n"); print(comparison_helpful)

###############################################
# END OF SCRIPT
###############################################


tbl
