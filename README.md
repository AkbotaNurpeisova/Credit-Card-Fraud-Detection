This project focuses on detecting fraudulent credit card transactions using various machine learning techniques. The dataset is highly imbalanced, with fraudulent transactions making up only 0.17% of all transactions. The goal is to build models that can effectively identify fraud while minimizing false positives.

The dataset contains credit card transactions made by European cardholders in September 2013. It includes 284,807 transactions, of which only 492 are fraudulent.

- `Time`: Seconds elapsed between this transaction and the first transaction
- `V1` through `V28`: PCA-transformed features (anonymized for privacy)
- `Amount`: Transaction amount
- `Class`: Target variable (1 = fraud, 0 = legitimate)

Key Objectives:
- Handle highly imbalanced dataset (0.17% fraud cases)
- Perform exploratory data analysis (EDA) to understand data patterns
- Apply SMOTE oversampling to balance the training data
- Compare multiple classification algorithms
- Build and optimize a Neural Network for fraud detection
- Evaluate models using appropriate metrics (precision, recall, F1-score, ROC-AUC)


Data Preprocessing:
- Feature Scaling: Applied RobustScaler to `Amount` and `Time` features (robust to outliers)
- Handling Imbalance: Used SMOTE (Synthetic Minority Over-sampling Technique) to balance the training dataset
  - Before SMOTE: 394 fraud transactions
  - After SMOTE: 227,451 fraud transactions (balanced with legitimate transactions)

Key Findings:
- Neural Network achieved the best performance with ROC-AUC close to 0.99, demonstrating strong ability to distinguish between fraud and legitimate transactions.
- Random Forest showed solid performance (~0.96 ROC-AUC) with good precision-recall balance for fraud detection.
- Logistic Regression performed reasonably well but struggled with precision for the minority class.
- Isolation Forest had lower performance as expected for unsupervised anomaly detection on this dataset.
