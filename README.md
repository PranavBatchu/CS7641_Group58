# Proposal



## Introuction

The problem we are trying ot solve is the unpredictability of sports betting on soccer games. We aim to solve this problem by leveraging ML to create a model that predicts the outcome of soccer matches. Our ultimate goal is that this model will serve as a useful tool for sports betting on soccer.

### Literature Review
The application of machine learning (ML) in sports analytics, particularly in soccer, has garnered significant attention in recent years. Various studies have explored predictive modeling techniques to forecast match outcomes, leveraging historical data and advanced algorithms. One paper utilized logistic regression models to analyze team performance metrics, revealing that specific statistics, such as possession and shots on target, are strong predictors of match results. Another proposed a hybrid model combining ensemble learning and neural networks to improve the prediction of soccer match outcomes. Despite these advancements, many models still struggle with the inherent unpredictability of sports events, which can be influenced by unpredictable factors like referee decisions, player behavior, and real-time game dynamics.

### Dataset Description
This dataset contains historical data of soccer matches. Each row represents a unique match with various features describing the teams' performance prematch, betting odds, and more. Each row is also labeled as a HomeWin, Draw, or AwayWin.

### Dataset Link
https://www.kaggle.com/competitions/prediction-of-results-in-soccer-matches/data  



## Problem Definition

### Problem
The primary problem we are addressing is the unpredictability and uncertainty associated with sports betting on soccer matches. Traditional methods of predicting match outcomes often rely on simplistic heuristics or historical win-loss records, which fail to account for the complex and dynamic nature of soccer games. This unpredictability can lead to substantial financial losses for bettors and may discourage participation in sports betting. As a result, there is a pressing need for a more sophisticated and data-driven approach to accurately forecast match results.

### Motivation
The motivation for this project stems from the growing interest in sports analytics and the potential economic benefits it can bring to both bettors and sports organizations. With the increasing availability of detailed match data, betting odds, and player statistics, machine learning offers a promising avenue to enhance prediction accuracy. By developing a model that effectively analyzes historical soccer match data, we aim to provide a valuable tool for sports bettors, enabling them to make informed decisions based on data rather than intuition. 


## Methods

### Data Preprocessing Methods Identified
The dataset provided is from a Kaggle competition so it is already cleaned, well-formatted, labeled, and split into training/testing subsets. Due to this, we do not see any heavy data preprocessing necessary beyond just parsing the provided csv files.

### ML Algorithms/Models Identified
Logistic Regression: As a foundational algorithm for binary classification, logistic regression will be employed to predict match outcomes (HomeWin, Draw, AwayWin). This model is straightforward to implement and interpret, making it a suitable choice for initial predictions based on performance metrics and betting odds.

Random Forest Classifier: To enhance prediction accuracy, we will utilize a Random Forest Classifier. By aggregating predictions from multiple decision trees, this model can better capture complex patterns in the data and reduce the risk of overfitting, leading to improved generalization on unseen data.

Gradient Boosting Classifier: Lastly, we will utilize a Gradient Boosting Classifier. This will help with understanding the relationships between the match features and correcting prior residual errors. It is a method that constructs weak learners in an organized manner, and with specific tuning of hyperparameters it can result in very accurate performance.

### Unsupervised and Supervised Learning Methods Identified
Even though our data is already labeled, we will explore clustering methods such as K-means clustering to identify patterns within teams based on their performance metrics. This unsupervised approach can provide insights into team behavior that may enhance our understanding of match dynamics.

However, the primary focus of our project will be on supervised learning methods, as our dataset contains labeled outcomes (HomeWin, Draw, AwayWin). By training our models on this labeled data, we can optimize them to make accurate predictions based on input features.



## Potential Results and Discussion

### Quantitative Metrics
- **Accuracy**: This metric will measure the overall correctness of our predictions by calculating the ratio of correctly predicted outcomes to the total number of predictions made. It will give us a general sense of how well our models perform across all classes (HomeWin, Draw, AwayWin).
- **Precision** and Recall: Precision will help us assess the accuracy of positive predictions (e.g., predicting a HomeWin), while recall will measure the ability of our model to identify all actual positive cases. Both metrics will be critical in understanding the trade-offs between correctly predicting match outcomes and minimizing false positives or negatives.
- **F1 Score**: The F1 Score, which is the harmonic mean of precision and recall, will be utilized to balance the trade-off between these two metrics. This will be particularly valuable in scenarios where class distribution is imbalanced, ensuring that our model remains robust in predicting less frequent outcomes.

### Project Goals
- We aim to achieve 75% or higher on all performance metrics
- We hope our model can promote financially responsible decions while sports betting

### Expected Results
- With the combination of logistic regression and Random Forest classifiers, we aim to achieve competitive accuracy metrics, with an F1 Score exceeding 0.75 for all classes, demonstrating effective model performance across different outcomes.
- Through the exploration of clustering methods, we expect to uncover hidden patterns in team performance that could inform strategic betting decisions and enhance our understanding of the underlying factors that influence match outcomes.


## References



## Gantt Chart

https://gtvault-my.sharepoint.com/:x:/g/personal/amistry31_gatech_edu/ERdTW3JDGQ9NibdAraEwbAIBa_T8pidvcs2EdB4nqX4twg?e=eV1qHs 


## Contribution Table

| Name     | Proposal Contributions |
|----------|------------------------|
| Ahaan    |                        |
| Akshay   |                        |
| Aryan    |                        |
| Pranav   |                        |
| Veer     |                        |

