# Importing libraries
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split

def model(Feature,Target):
    Feature_train,Feature_test,Target_train,Target_test=train_test_split(Feature, Target, test_size=0.50, random_state=1)

    # instantiate the model (using the default parameters)
    LogisticModel = LogisticRegression()

    # fit the model with data
    LogisticModel.fit(Feature_train,Target_train)
    Target_pred = LogisticModel.predict(Feature_test)
    # print Accuracy
    return print("Accuracy:",metrics.accuracy_score(Target_test, Target_pred))


data = pd.read_csv("papajohns.csv")
#data.fillna(0, inplace=True)

# Convert float to true or false for make_time
# If the make time less than 15 min then it's true and on time otherwise false
newmaketime = ([i < 15 for i in data.make_time])
# Convert true or false to 1 or 0 for make_time
newmaketime = list(map(int, newmaketime))
#print(newmaketime)

#split dataset in features and target variable

allfeatures = ['quantity_sold', 'is_plan_ahead', 'stuffed_crust', 'inhopper']

# Features
Features = data[allfeatures]
# Target variable
Target = newmaketime

#results
model(Features, Target)



