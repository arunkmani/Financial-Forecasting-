import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
def compare(x):
    if x> -0.5:
        return 0
    else:
        return 1
df= pd.read_csv('fin.csv')
df= df.drop('Company', axis=1)
df= df.drop('Time', axis=1)
temp= df['Financial Distress'].apply(compare)
df['Financial Distress'] = temp
normalized_df=(df-df.mean())/df.std()

X= df.loc[:,df.columns !='Financial Distress']
y= df['Financial Distress']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_predictor = model.predict(X_test)
accuracy = accuracy_score(y_test,y_predictor)
print('Accuracy isttru:',accuracy)
report = classification_report(y_test, y_predictor)
print(report)