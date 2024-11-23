# Machine-Learning-Models
Implementation of traditional ML models


## 1. Classifiers
Classifiers predict **discrete outcomes or categories**. Common use cases include spam detection, disease diagnosis, and image classification.

### Popular Classification Models
1. **Logistic Regression**:
   - Linear model for binary or multinomial classification.
   - Outputs probabilities for each class.

2. **K-Nearest Neighbors (KNN)**:
   - Predicts the class based on the majority class of its nearest neighbors.
   - Best for smaller datasets.

3. **Support Vector Machine (SVM)**:
   - Separates classes with a hyperplane.
   - Kernel trick enables handling non-linear separations.

4. **Decision Trees**:
   - Rule-based model; splits data based on feature thresholds.
   - Easy to interpret but prone to overfitting.

5. **Random Forest**:
   - Ensemble of decision trees; reduces overfitting and improves generalization.
   - Handles categorical and continuous data effectively.

6. **Gradient Boosting Machines (e.g., XGBoost, LightGBM)**:
   - Builds models sequentially to correct previous errors.
   - High accuracy but computationally intensive.

7. **Naive Bayes**:
   - Probabilistic classifier based on Bayes' theorem.
   - Assumes feature independence. Effective for text classification.

8. **Neural Networks**:
   - Includes feedforward and convolutional networks.
   - Suitable for high-dimensional data like images and audio.

9. **k-Means (Semi-Supervised)**:
   - Often used for clustering but can assist in classification tasks.

### Classification Metrics
- **Accuracy**: Proportion of correctly predicted instances.
- **Precision**: True positives divided by all predicted positives.
- **Recall (Sensitivity)**: True positives divided by all actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the ROC curve; measures the ability to distinguish classes.

---

## 2. Regressors
Regressors predict **continuous numerical values**. Common use cases include predicting house prices, stock prices, and weather forecasts.

### Popular Regression Models
1. **Linear Regression**:
   - Predicts output as a linear combination of inputs.
   - Simple and interpretable but limited to linear relationships.

2. **Ridge and Lasso Regression**:
   - Regularized versions of linear regression.
   - **Ridge** penalizes large coefficients (L2 norm).
   - **Lasso** enforces sparsity (L1 norm).

3. **Polynomial Regression**:
   - Extends linear regression with polynomial features.
   - Suitable for non-linear relationships.

4. **K-Nearest Neighbors (KNN)**:
   - Predicts by averaging the values of the nearest neighbors.
   - Simple but computationally expensive.

5. **Support Vector Regressor (SVR)**:
   - Fits a margin around true values instead of predicting exact points.
   - Extension of SVM for regression tasks.

6. **Decision Trees**:
   - Splits data into intervals for prediction.
   - Tends to overfit unless regularized.

7. **Random Forest Regressor**:
   - Ensemble of decision trees; reduces variance and improves predictions.
   - Works well for complex data distributions.

8. **Gradient Boosting Machines (e.g., XGBoost, LightGBM, CatBoost)**:
   - Builds models sequentially, optimizing errors of prior iterations.
   - High accuracy and widely used in competitions.

9. **Neural Networks**:
   - Multilayer perceptrons or deep architectures for complex regression problems.
   - Requires large datasets and computational power.

10. **Gaussian Processes**:
    - Bayesian approach to regression.
    - Provides uncertainty estimates with predictions.

### Regression Metrics
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values.
- **Mean Squared Error (MSE)**: Penalizes larger errors by squaring them.
- **Root Mean Squared Error (RMSE)**: Square root of MSE; same units as the target.
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model.

---

## Choosing Between Classifiers and Regressors
- **Type of Output**:
  - Use classifiers for discrete categories (e.g., yes/no, red/blue).
  - Use regressors for continuous values (e.g., 1.5, 100.3).
- **Model Complexity**:
  - Start with simple models like linear models or decision trees for interpretability.
  - Use ensemble methods or neural networks for high-dimensional, complex data.
