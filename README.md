#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


calories0=pd.read_csv('/content/calories.csv')
exercise_data=pd.read_csv('/content/exercise.csv')
calories0.head()
exercise_data.head()
calories_data=pd.concat([exercise_data,calories0['Calories']],axis=1)
calories_data.head()
calories_data.size
calories_data.shape
calories_data.info()


#now finding some statistical data about the calories
calories_data.describe()
sns.set()

#to print and compare the different data
sns.countplot(calories_data['Gender'])

# Calculate correlation only for numerical columns
correlation = calories_data.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8}, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()

calories_data.replace({"Gender":{'male':0,'female':1}},inplace=True)
x=calories_data.drop(columns=['User_ID','Calories'], axis=1)
y=calories_data['Calories']
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)
model=XGBRegressor()
model.fit(x_train,y_train)
test_data_prediction=model.predict(x_test)
print(test_data_prediction)
