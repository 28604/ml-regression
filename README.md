# ml-regression

Regression algorithm using linear models coded from scratch.

### Problem: use different regression approaches to predict altitudes of southern Taiwan

<img src="https://github.com/28604/ml-regression/blob/main/img/altitude.png" width="600" alt="An image of southern taiwan 3D topographic map">
<img src="https://github.com/28604/ml-regression/blob/main/img/regression%20using%20linear%20model.png" width="600" alt="An image of regression model pipeline">
### Description:
The training dataset contains 6,000 pairs of coordinates and their corresponding altitude values. You should create feature vectors based on the input coordinates as the model’s inputs and train your regression models to fit the training data. Predict the altitude values for the 2,000 coordinates in the testing dataset and save your predictions along with the model’s weights as a .csv file.
For each approach, the **mean squared error (MSE) of your predictions on the testing dataset must be less than 900**; otherwise, you will fail the correctness check.

### Grading Policy
After passing the correctness check, TAs will evaluate your performance based on the results of your three approaches and select the best one for ranking. Your performance score will be calculated using the following formula: 

$Performance\ Score = Mean\ Squared\ Error \times Number\ of\ Weights$

Your final score in this part will be determined based on the ranking of your performance score.

### Regression Methods:
* Maximum Likelihood <br/>
  <img src="https://github.com/28604/ml-regression/blob/main/img/maximum%20likelihood.png" width="400" alt="An image of maximum likelihood formula">
* Maximum A Posteriori <br/>
  <img src="https://github.com/28604/ml-regression/blob/main/img/maximum%20a%20posteriori.png" width="400" alt="An image of maximum a posteriori formula"><br/>
  <img src="https://github.com/28604/ml-regression/blob/main/img/maximum%20a%20posteriori%20methods.png" width="400" alt="An image of maximum a posteriori formula">
* Bayesian Regression Method <br/>
  <img src="https://github.com/28604/ml-regression/blob/main/img/bayesian%20regression%20method.png" width="400" alt="An image of bayesian regression formula">

### Other Details
* K-means to find cluster centroids for Gaussian basis function
* K-fold cross validation to avoid overfitting
