# Fake Review Detection & Helpfulness Prediction on Yelp

**Master Thesis** – Data Science & Marketing Analytics  
**Erasmus School of Economics** – October 2025  
**Author:** Anastasis Gkoutas  
**Supervisor:** dr. Michel van de Velden

## 📌 Project Overview
End-to-end R pipeline that detects **fake reviews** and predicts **helpfulness** using 27,000+ real Yelp restaurant reviews.

The project uses exactly the same models and features as my Master's thesis:
- LASSO Logistic Regression
- Random Forest (vanilla + balanced)
- **XGBoost** (best overall performance)

## 📊 Key Results (from the thesis)

### Fake Review Detection
| Model                  | Accuracy | Precision | Recall | F1    | AUC  |
|------------------------|----------|-----------|--------|-------|------|
| LASSO                  | 0.77     | 0.53      | 0.08   | 0.13  | 0.68 |
| Random Forest          | 0.78     | 0.59      | 0.06   | 0.10  | 0.70 |
| Balanced RF            | 0.78     | 0.59      | 0.06   | 0.10  | 0.70 |
| **XGBoost**            | **0.78** | **0.60**  | **0.12**| **0.20**| **0.71** |

### Helpfulness Prediction
| Model                  | Accuracy | Precision | Recall | F1    | AUC  |
|------------------------|----------|-----------|--------|-------|------|
| LASSO                  | 0.73     | 0.61      | 0.25   | 0.35  | 0.73 |
| Random Forest          | 0.73     | 0.64      | 0.24   | 0.35  | 0.72 |
| Balanced RF            | 0.73     | 0.64      | 0.24   | 0.34  | 0.72 |
| **XGBoost**            | **0.73** | 0.60      | **0.32**| **0.42**| **0.74** |

**Conclusion:** XGBoost performs best overall. Helpfulness is much easier to predict than deception (fake reviews look very similar to real ones).

## ✨ Features Used
- Sentiment polarity & subjectivity (using `sentimentr`)
- Readability (Automated Readability Index)
- Review depth (log-transformed word count)
- TF-IDF (206 terms after sparsity filtering)
- Star rating + composite helpfulness score

## 📂 How to Run
```r
source("yelp-fake-review-helpfulness.R")
