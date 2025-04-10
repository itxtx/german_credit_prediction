# German Credit Risk Model Comparison Report

Date:  April 10, 2025 

## 1. Overview

This report compares the performance of different machine learning models trained on the German Credit Risk dataset. The models are evaluated on various metrics to determine which one performs best for predicting credit risk.

## 2. Models Compared

The following models were trained and evaluated:

- ** xgboost **
- ** random forest **
- ** logistic regression **
- ** decision tree **
- ** svm **

## 3. Performance Metrics

### 3.1 Comparison Table

| Model |  accuracy | precision | recall | f1 | auc  |
| ---|---|---|---|---|--- |
|  xgboost  |  **0.6767** | 0.8383 | **0.6667** | **0.7427** | **0.7761**  |
|  random_forest  |  0.6667 | **0.8618** | 0.6238 | 0.7238 | 0.7743  |
|  logistic_regression  |  0.66 | 0.8418 | 0.6333 | 0.7228 | 0.7575  |
|  decision_tree  |  0.6067 | 0.8239 | 0.5571 | 0.6648 | 0.684  |
|  svm  |  0.5333 | 0.7612 | 0.4857 | 0.593 | 0.6007  |

## 4. Best Model

Based on the  auc  metric, the best performing model is ** xgboost ** with a value of ** 0.7761 **.

### 4.1 Model Ranking

| Rank | Model |  auc  |
|---|---|---|
|  1  |  xgboost  |  0.7761  |
|  2  |  random forest  |  0.7743  |
|  3  |  logistic regression  |  0.7575  |
|  4  |  decision tree  |  0.684  |
|  5  |  svm  |  0.6007  |

## 6. Performance Visualizations

### 6. 1   A ccuracy  Comparison

![ A ccuracy  Comparison]( accuracy _comparison.png)

### 6. 2   P recision  Comparison

![ P recision  Comparison]( precision _comparison.png)

### 6. 3   R ecall  Comparison

![ R ecall  Comparison]( recall _comparison.png)

### 6. 4   F 1  Comparison

![ F 1  Comparison]( f1 _comparison.png)

### 6. 5   A uc  Comparison

![ A uc  Comparison]( auc _comparison.png)

### 6. 6  All Metrics Comparison

![All Metrics Comparison](all_metrics_comparison.png)

### 6. 7  Radar Chart

![Radar Chart](radar_chart.png)

## 7. Conclusion

After comparing various machine learning models on the German Credit Risk dataset, the ** xgboost ** model demonstrated the best performance with a  auc  of  0.7761 . This suggests that ensemble methods perform well on this dataset, likely due to their ability to handle complex relationships and interactions between features.

For credit risk prediction tasks on similar data, the  xgboost  model is recommended based on its superior performance.

