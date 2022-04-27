import csv, numpy as np, pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score



DAYPART = {"Afternoon": 1, "Dinner": 3, "Evening": 2, "Late Night": 4, "Lunch": 0}
DESTTYPE = {"Carryout": 0, "Delivery": 1}
SRCCHAN = {"App": 0, "Phone": 1, "Store": 2, "Web": 3}
COLUMNS = {"make_time": 17, "otd_time": 21, "day_part": 60,
           "source_channel": 69, "destination_type": 71} # "bake_time": 19, "rack_time": 20

# read data
infile = open("Updated_PJ_Orders.csv", 'r')
incsv = csv.reader(infile)
data = []
for row in incsv:
    data.append(row)
data.pop(0) # get rid of title row

include_indices = list(COLUMNS.values())

# include only columns listed above
for row in data:
    for i in range(len(row) - 1, -1, -1): # count down from last index to 0
        if i not in include_indices:
            row.pop(i)

# replace non-numerics with mapped integers and numerics with floats
# also remove entries with null values in any newcolumns field
new_columns = {k:list(COLUMNS.values()).index(v) for k,v in COLUMNS.items()}
to_remove = []
for j in range(len(data)):
    for i in range(5):
        if data[j][i] == '':
            to_remove.append(j)
            break
        if i < 2:
            data[j][i] = float(data[j][i])
        else:
            data[j][i] = list(new_columns.values())[i]
    data[j] = np.array(data[j])
to_remove.reverse()
for ind in to_remove:
    data.pop(ind)
print(data[0])


# define training and test sets
# training: 70%
# test: 30%
training_set = []
test_set = []

for i in range(len(data)):
    if i < ((len(data) * 7) // 10):
        training_set.append(data[i])
    else:
        test_set.append(data[i])
# training_set = np.array(training_set)
# test_set = np.array(test_set)
# print(f"training: {training_set.size}\ntest: {test_set.size}")

# train the model
reg = linear_model.LinearRegression()
X, xTest = [], [] # other features
y = [] # otd_time
yTest = []
for row in training_set:
    X.append(np.delete(row, 1))
    y.append(row[1])
for row in test_set:
    xTest.append(np.delete(row, 1))
    yTest.append(row[1])

# X = pd.DataFrame(X)
# y = pd.DataFrame(y)
Xn = np.array(X)
yn = np.array(y)

# reg.fit(X, y)
reg.fit(Xn, yn)
print(reg.coef_, reg.intercept_, "\n")
predictions = reg.predict(xTest)

for i in range(5):
    print(f"point: {xTest[i]} ->> {predictions[i]}")

output = open("linreg_model.csv", 'w')
outcsv = csv.writer(output)
for i in range(len(xTest)):
    row = xTest[i].tolist() + [predictions[i].tolist()]
    outcsv.writerow(row)


print(f"r2score = {r2_score(yTest, predictions)}\nMSE = { mean_squared_error(yTest, predictions)}")

output.close()