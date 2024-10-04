# Predict_Weight-loss
The object of this code is to predict the weight loss based on the number of hours a person exercise using a simple linear Regression model.

Here’s an explanation of every line in the code you shared:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import pandas as pd: Imports the pandas library and assigns it the alias pd, which is used to handle and manipulate data in tabular form.

from sklearn.model_selection import train_test_split: Imports the train_test_split function from the sklearn.model_selection module. This function helps split data into training and testing sets.

from sklearn.linear_model import LinearRegression: Imports the LinearRegression class from the sklearn.linear_model module, which will be used to create the linear regression model.

import matplotlib.pyplot as plt: Imports matplotlib.pyplot as plt, a library used for data visualization.


# Create a dictionary with example data for Exercise Hours and Weight Loss
data_dict = {
    'Exercise_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Weight_Loss': [2, 3, 5, 7, 8, 9, 11, 13, 14]
}

data_dict: A dictionary is created with two keys, 'Exercise_Hours' and 'Weight_Loss'. Each key contains a list of values representing the number of exercise hours and corresponding weight loss, respectively.


# Convert the dictionary to a DataFrame
data = pd.DataFrame(data_dict)

pd.DataFrame(data_dict): The dictionary data_dict is converted into a DataFrame using pandas. This structure organizes the data into a tabular format with labeled rows and columns, making it easier to manipulate and analyze.


# Display the data in table format
print("Data from Dictionary:")
print(data)

print("Data from Dictionary:"): Prints a header string to indicate that the table will be displayed.

print(data): Prints the data DataFrame (which shows the Exercise Hours and Weight Loss columns) in a table format.


# Prepare the features (Exercise Hours) and target variable (Weight Loss)
X = data[['Exercise_Hours']]
y = data['Weight_Loss']

X = data[['Exercise_Hours']]: Creates X, the feature set, by selecting the column Exercise_Hours from the DataFrame. It's stored as a 2D array because LinearRegression requires features in this format.

y = data['Weight_Loss']: Creates y, the target variable, by selecting the Weight_Loss column from the DataFrame. This is the value that the model will attempt to predict.


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_test_split: Splits the data into training and testing sets.

X_train and y_train: These will be used to train the model (80% of the data).

X_test and y_test: These will be used to test the model's performance (20% of the data).

test_size=0.2: Specifies that 20% of the data will be used for testing.

random_state=42: Ensures that the split is reproducible by setting a seed.



# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

model = LinearRegression(): Initializes a linear regression model object.

model.fit(X_train, y_train): Trains (fits) the model using the training data (X_train and y_train). The model learns the relationship between the features (exercise hours) and the target variable (weight loss).


# Take user input for exercise hours
user_input = float(input("Enter the number of exercise hours to predict weight loss: "))

input(): Prompts the user to input a value, specifically the number of exercise hours.

float(): Converts the user's input from a string to a floating-point number (decimal), which is necessary for numerical calculations.


predicted_weight_loss = model.predict([[user_input]])

model.predict([[user_input]]): Uses the trained model to predict the weight loss for the provided user_input (number of exercise hours). The input must be a 2D array (i.e., [[user_input]]), as predict expects it in this format.

predicted_weight_loss: Stores the predicted weight loss as a result of the predict function.


# Display the predicted weight loss
print(f"Predicted weight loss for {user_input} hours of exercise: {predicted_weight_loss[0]} kg")

print(f"...{predicted_weight_loss[0]}..."): Prints a message showing the user’s input (exercise hours) and the predicted weight loss. predicted_weight_loss[0] extracts the first (and only) predicted value from the array.


# Plot the data points and the regression line
plt.scatter(X, y, color='blue', label='Actual Weight Loss')
plt.plot(X, model.predict(X), color='red', label='Fitted Line')
plt.xlabel('Exercise Hours')
plt.ylabel('Weight Loss (kg)')
plt.title('Exercise Hours vs Weight Loss')
plt.legend()
plt.show()

plt.scatter(X, y, color='blue', label='Actual Weight Loss'): Plots the actual data points as a scatter plot. X (Exercise Hours) is on the x-axis and y (Weight Loss) on the y-axis. The points are colored blue and labeled 'Actual Weight Loss'.

plt.plot(X, model.predict(X), color='red', label='Fitted Line'): Plots the linear regression line (fitted by the model) in red. It uses the predict method to get the predicted values for the training data (X).

plt.xlabel('Exercise Hours'): Sets the x-axis label to 'Exercise Hours'.

plt.ylabel('Weight Loss (kg)'): Sets the y-axis label to 'Weight Loss (kg)'.

plt.title('Exercise Hours vs Weight Loss'): Sets the title of the plot.

plt.legend(): Adds a legend to differentiate between the actual data points and the fitted line.

plt.show(): Displays the plot.


In summary, this code performs data visualization, model training, prediction, and plots the regression line to show the relationship between exercise hours and weight loss.
