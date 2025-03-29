# German Credit Risk Model Comparison Report

Date:  March 29, 2025 

## 1. Overview

This report compares the performance of different machine learning models trained on the German Credit Risk dataset. The models are evaluated on various metrics to determine which one performs best for predicting credit risk.

## 2. Models Compared

The following models were trained and evaluated:

- ** random forest **
- ** logistic regression **
- ** naive bayes **

## 3. Performance Metrics

### 3.1 Comparison Table

| Model |  accuracy | precision | recall | f1 | auc  |
| ---|---|---|---|---|--- |
|  random_forest  |  **0.6667** | **0.8618** | 0.6238 | **0.7238** | **0.7743**  |
|  logistic_regression  |  0.66 | 0.8418 | **0.6333** | 0.7228 | 0.7575  |
|  naive_bayes  |  0.6 | 0.8409 | 0.5286 | 0.6491 | 0.6979  |

## 4. Best Model

Based on the  auc  metric, the best performing model is ** random forest ** with a value of ** 0.7743 **.

### 4.1 Model Ranking

| Rank | Model |  auc  |
|---|---|---|
|  1  |  random forest  |  0.7743  |
|  2  |  logistic regression  |  0.7575  |
|  3  |  naive bayes  |  0.6979  |

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

After comparing various machine learning models on the German Credit Risk dataset, the ** random forest ** model demonstrated the best performance with a  auc  of  0.7743 . This suggests that ensemble methods perform well on this dataset, likely due to their ability to handle complex relationships and interactions between features.

For credit risk prediction tasks on similar data, the  random forest  model is recommended based on its superior performance.

