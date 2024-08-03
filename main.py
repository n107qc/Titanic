# Тут має бути твій код
import pandas as pd 
df = pd.read_csv("titanic.csv")
df.info()
print(df.groupby('Sex')['Survived'].mean())
print(df.pivot_table(index='Survived',columns='Pclass',values='Age',aggfunc='mean'))
print(df.groupby("Pclass")['Survived'].mean())
print(df.groupby("SibSp")['Survived'].mean())
print(df.groupby("Parch")['Survived'].mean())
print(df.groupby("Embarked")['Survived'].mean())


df.info()

def fill_sex(data):
    if data == 'male':
        return 1
    else:
        return 0 
    
df["Sex"] = df['Sex'].apply(fill_sex)
df.info()

age1 = df[df['Pclass'] == 1]['Age'].median()
age2 = df[df['Pclass'] == 2]['Age'].median()
age3 = df[df['Pclass'] == 3]['Age'].median()

def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return age1
        if row['Pclass'] == 2:
            return age2
        if row['Pclass'] == 3:
            return age3
    return row['Age']

df['Age'] = df.apply(fill_age, axis=1)

df.info()
print(df['Embarked'].value_counts())
df['Embarked'].fillna('S',inplace=True)
df.info()

df[list(pd.get_dummies(df['Embarked']).columns)]=pd.get_dummies(df['Embarked'])
df.info()


df.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)
df.info()

def is_alone(data):
    if data['SibSp']+ data['Parch'] == 0:
        return 1
    return 0 

df['Alone'] = df.apply(is_alone, axis = 1)
df.info()

df.to_csv('cleaned_titanic.csv', index=False)