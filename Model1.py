import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('titanic.csv')    #to read the file
print(df)



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

##Replacing Null
df['Age'] = df['Age'].replace(np.nan, 0)
df['Embarked'] = df['Embarked'].replace(np.nan, 0)

print(df)


x=df.drop(columns=['Survived','Name','PassengerId','Ticket','Cabin'])
y=df['Survived']      #to create the variable

print("XXXX",x)
print("YYYY",y)




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)   #split the val

##X_train  - 80% input data
##Y_train  - 80% output data
##X_test   - 20% input data
##Y_test   - 20% output data


print("DF",df.shape)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)




from sklearn.naive_bayes import GaussianNB  
NB = GaussianNB()

NB.fit(x_train, y_train)


###train the data
y_pred=NB.predict(x_test)
print("y_pred",y_pred)
print("y_test",y_test)



from sklearn.metrics import accuracy_score
print('ACCURACY is', accuracy_score(y_test,y_pred))







testPrediction = NB.predict([[0,3,1,22.0,1,6,2]])
if testPrediction==1:
    print("Survide")
else:
    print("Not survied")


