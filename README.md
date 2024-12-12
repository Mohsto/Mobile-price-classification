# **Mobile Price Classification**

This project focuses on predicting the price range of mobile phones based on their technical specifications, such as RAM, battery power, screen size, 
and camera quality. The goal is to classify mobile phones into four price categories: low cost, medium cost, high cost, and very high cost.

# **Data Source**
- The dataset contains 2,000 records and 21 features, including attributes like RAM, battery power, camera specifications, and more.
- The dataset used in this project is synthetic and designed for mobile price classification.
- Dataset from Kaggle: [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv)

# **Technologies Used**
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Jupyter Notebook**

# **Key Insights**
- **Battery Power & RAM**: Higher RAM and battery power are strongly associated with higher price ranges.
- **Pixel Dimensions**: Pixel dimensions (height and width) are significant indicators of price, with larger dimensions linked to higher prices.
- **Screen Size**: A larger screen size tends to correlate with higher-end phones, especially in the price_range classes 2 and 3.
- **Model Performance**: Random Forest emerged as the best-performing model, achieving an accuracy of 89.00% and a weighted F1 score of 89.04%.

## **Modeling Approach**
- **Feature Engineering**: New features were created, such as:
- **Pixels**: Product of Pixel Height and Pixel Width.
- **Diagonal**: The screen size calculated using the Pythagorean theorem.
- **Data Preprocessing**: The dataset was transformed using normalization, and standardization techniques.
- **Algorithms**:
  - Logistic Regression (Baseline)
  - Random Forest (Best Performer)
  - K-Nearest Neighbors (KNN)
  - Naive Bayes

- **Evaluation Metrics**: 
  - Accuracy, Precision, Recall, F1-Score

| **Model**           | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
|---------------------|--------------|---------------|------------|--------------|
| Logistic Regression | 82.62%       | 82.09%        | 82.62%     | 82.14%       |
| Random Forest       | **86.62%**   | **86.74%**    | **86.62%** | **86.60%**   |
| KNN                 | 39.69%       | 41.30%        | 39.69%     | 39.52%       |
| Naive Bayes         | 80.06%       | 80.43%        | 80.06%     | 80.13%       |

## **How to Run the Analysis**
1. Download the dataset from Kaggle.
2. Install the required Python libraries.
3. Run the `Mobile_price_classification.ipynb` notebook to replicate the analysis and predictions.

## **Future Enhancements**
- Explore other machine learning models like Gradient Boosting or Neural Networks.
- Experiment with hyperparameter tuning for Random Forest to improve model performance.
- Use advanced explainability techniques such as SHAP to interpret model predictions.
- Deploy the model as a web app for real-time price range predictions.
