
# Predict Calorie Expenditure

## Overview
This project develops a machine learning model to predict calorie expenditure during exercise, based on the Kaggle Playground Series - Season 5, Episode 5 dataset. The model leverages features such as weight, exercise intensity, duration, age, and gender to provide accurate predictions, enabling personalized fitness and health insights.
Value Proposition: The model delivers precise, data-driven calorie expenditure estimates, empowering users to optimize workout plans, manage weight, and achieve health goals. It has potential applications in fitness apps, wearable devices, and health coaching platforms, promoting sustainable lifestyle changes.
Dataset
The dataset is sourced from Kaggle's Predict Calorie Expenditure competition. It includes features such as:

**Weight**: Body weight in kilograms.  
**Exercise Intensity**: Metrics like heart rate.  
**Duration**: Time spent exercising in minutes.  
**Age**: Age of the individual.  
**Gender**: Male or female.  
**Height**: Height in centimeters.  
**Target**: Calorie expenditure (in kcal).

The dataset is preprocessed to handle missing values, encode categorical variables, and create interaction terms for improved model performance.
Methodology

**Data Preprocessing**: Handled missing data, scaled numerical features, and encoded categorical variables i.e gender.  
**Feature Engineering**: Created interaction terms (e.g., weight * intensity) and used permutation importance to identify high-impact features.  
**Model Selection**: Trained and evaluated models including Random Forest, XGBoost, and Linear Regression, with hyperparameter tuning via GridSearchCV.
Evaluation: Used RMSE (Root Mean Squared Error) to measure performance on the test set, achieving an RMSE of [11.0975, 7.559, 7.554].
Technologies: Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Jupyter Notebook, seaborn.

Installation

Clone the repository:git clone https://github.com/bamanyeronald/Predictive-Model-to-predict-Calorie-expenditure-with-different-algorithms


Navigate to the project directory:cd calorie-expenditure-prediction


Install dependencies:pip install -r requirements.txt


Ensure the dataset (train.csv, test.csv) is placed in the data/ folder.

Usage

Run the main script to train and evaluate the model:python src/main.py


Explore the Jupyter Notebook (notebooks/exploration.ipynb) for data analysis and model experimentation.
View model predictions and feature importance plots in the outputs/ folder.

Results

Achieved an RMSE of [insert metric, e.g., 25.4 kcal] on the test set using XGBoost.
Key features (weight, exercise intensity, duration) were identified as the most predictive, aligning with physiological principles.
The model generalizes well across diverse user profiles, suitable for real-world fitness applications.

Future Work

Incorporate additional features (e.g., body fat percentage, exercise type) for enhanced accuracy.
Deploy the model as an API for integration with fitness apps or wearables.
Explore deep learning models (e.g., neural networks) for complex feature interactions.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, feature additions, or documentation improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or collaboration, reach out via your.bamanyeronald@gmail.com or open an issue on GitHub.
