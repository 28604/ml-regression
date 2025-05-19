# ml-regression

Regression algorithm using linear models coded from scratch.

### Problem: use different regression approaches to predict altitudes of southern Taiwan

<img src="https://github.com/28604/ml-regression/blob/main/altitude.png" width="600" alt="An image of southern taiwan 3D topographic map">

### Description:
The training dataset contains 6,000 pairs of coordinates and their corresponding altitude values. You should create feature vectors based on the input coordinates as the model’s inputs and train your regression models to fit the training data. Predict the altitude values for the 2,000 coordinates in the testing dataset and save your predictions along with the model’s weights as a .csv file.
For each approach, the **mean squared error (MSE) of your predictions on the testing dataset must be less than 900**; otherwise, you will fail the correctness check.

### Grading Policy
After passing the correctness check, TAs will evaluate your performance based on the results of your three approaches and select the best one for ranking. Your performance score will be calculated using the following formula: 

$Performance\ Score = Mean\ Squared\ Error \times Number\ of\ Weights$

Your final score in this part will be determined based on the ranking of your performance score.

### Regression Methods:
* Maximum Likelihood
  
* Maximum A Posteriori
  
* Bayesian Regression Method
  

### Other Details
* K-means to find cluster centroids for Gaussian basis function
* K-fold cross validation to avoid overfitting
