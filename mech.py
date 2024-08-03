import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df  = pd.read_csv('cleaned_titanic.csv')
x = df.drop('Survived',axis=1)
y = df["Survived"] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

id_test = x_test["PassengerId"]
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_predict = knn.predict(x_test)
 
accuracy = accuracy_score(y_test,y_predict) * 100
print("Точність передбачення:",accuracy)



result = pd.DataFrame({'id':id_test,"predict":y_predict,'real_result':y_test})
result.to_csv('predict.csv',index=False)