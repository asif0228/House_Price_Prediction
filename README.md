# House_Price_Prediction
House Price Prediction using Artificial Intelligence 

Development Tech:
- Python

This is a Learning Poject. Purpose is to predict House price using different AI techniques. Starting from most simple Batch Gradient Descent.

This project is inspired by coursea course Predicting House Prices with Regression using TensorFlow.
Link: https://www.coursera.org/learn/tensorflow-beginner-predicting-house-prices-regression

Where as the course explicitly used Tensoflow. This version is more raw in nature. All the functions are written manually with sheer determination of learning .

Steps are as followed:


00. Main Function
01. Load data.csv and remove 1st column which is serial
02. Check for missing data
03. Normalize data, formula df_norm = (df - df.mean())/df.std() Here, std = standard deviation
04. as we normalized result (y) also so need a function to convert back our pridected result (y) to real y
05. Seperate x and y.
06. Seperate Training and Test set
07. Create NN Model
- 07.01 Initilize and set weights
08. Tain The Neural Netwok
- 08.01 Start Training NN for given iteration
- 08.02 Compute Cost
- 08.03 Update Weight Using Batch Gradient Descent
09. Plot cost vs iteration gaph
10. Predict using the trained Model