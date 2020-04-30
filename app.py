import streamlit as st
import pandas as pd
import numpy as np
import glob
import re 
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

st.title('Titanic survival') 




@st.cache
def load_data():
    import pandas as pd
    import glob

    csv1= pd.read_csv('https://github.com/lydiahiba/Data-IA-exercices-/raw/master/test.csv', index_col=None, header=0)
    csv2=pd.read_csv('https://github.com/lydiahiba/Data-IA-exercices-/raw/master/train.csv', index_col=None, header=0)

    titanic = pd.concat([csv1,csv2], axis=0, ignore_index=True)
    return titanic 

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1) ## to extract only the subgroup from the matching which is the mrs et miss ( to se the diference try without)
    return ""

def Preprocessing_data():

    csv1= pd.read_csv('https://github.com/lydiahiba/Data-IA-exercices-/raw/master/test.csv', index_col=None, header=0)
    csv2=pd.read_csv('https://github.com/lydiahiba/Data-IA-exercices-/raw/master/train.csv', index_col=None, header=0)

    titanic = pd.concat([csv1,csv2], axis=0, ignore_index=True)

    titanic['Sex']=titanic['Sex'].astype('category')

    titanic['Name']=titanic['Name'].astype('category')

    titanic['Embarked']=titanic['Embarked'].astype('category')
    titanic['Has_Cabin']=titanic.Cabin.apply(lambda x: 0 if pd.isnull(x) else 1)
    #drop the nan from age 
    age_serie= titanic[['Age']] 
    survived_serie=titanic.Survived
    imputer = KNNImputer()
    age_serie = imputer.fit_transform(age_serie)
    titanic['Age']=age_serie
    # convert age to categorical
    titanic['Age'] = titanic['Age'].astype(int)
    titanic['CategoricalAge'] = pd.cut(titanic['Age'], 5,labels=False)
    # Création une variable Name_length qui contient la longueur de la variable Name , askip plus le noms est long plus la perssone fais partie de la haute société et donc a plus de chance de survie 
    titanic['Name_length']=titanic['Name'].apply(len)
    # create a family column 
    titanic['FamilySize']=titanic.SibSp +titanic.Parch+1

    # Create new feature IsAlone from FamilySize
    titanic['IsAlone'] = 0
    titanic.loc[titanic['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    titanic['Embarked'] = titanic['Embarked'].fillna('S')

    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
    titanic['CategoricalFare'] = pd.qcut(titanic['Fare'], 4,labels=False)
    # Define function to extract titles from passenger names
    


    # Group all non-common titles into one single grouping "Rare"
    # Create a new feature Title, containing the titles of passenger names
    titanic['Title'] = titanic['Name'].apply(get_title)
    
    titanic['Title'] = titanic['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    titanic['Title'].unique()

    titanic['Title'] = titanic['Title'].replace('Mlle', 'Miss')
    titanic['Title'] = titanic['Title'].replace('Ms', 'Miss')
    titanic['Title'] = titanic['Title'].replace('Mme', 'Mrs')
    # Mapping Sex
    titanic['Sex'] = titanic['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    titanic.rename(columns={'Sex':'Male'},inplace=True)


    # Mapping titles
    titles= pd.get_dummies(titanic['Title'], drop_first=True)
    titanic = titanic.drop('Title', axis=1)
    titanic = titanic.join(titles)
    # Mapping Embarked
    titanic['Embarked'] = titanic['Embarked'].cat.codes
    titanic['Embarked'].unique()

    # Feature selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch','Fare','Age']
    titanic = titanic.drop(drop_elements, axis = 1)
    return titanic



# Load 10,000 rows of data into the dataframe.
data_prepro = Preprocessing_data()

def main():
    df= Preprocessing_data()
    gender =  st.sidebar.selectbox("Choose Sex :",['Male','Female'])

    if gender == "Male":
        st.write(df[df['Male']==1])
    
    elif gender == "Female":
        female=df[df['Male']==0]
        st.write(female)  

def  cabine():
    df= Preprocessing_data()
    cabineee =  st.sidebar.selectbox(" Does he have a Cabin:",['Yes','No'])

    if cabineee == "Yes":
        st.write(df[df['Has_Cabin']==1])
    
    elif cabineee == "No":
        st.write(df[df['Has_Cabin']==0])  
cabine()

#df= Preprocessing_data()
#sex= 
#gender =  st.sidebar.selectbox("Choose Sex :",df['Male'].unique().tolist())


if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

if st.checkbox('Show preproccessed data'):
    st.subheader('Preprocessed data')
    st.write(data_prepro)



test_set= titanic[titanic['Survived'].isnull()] 
train_set= titanic[titanic['Survived'].notna()]

    
X = train_set.drop(['Survived'], axis=1).values
y= train_set.Survived.values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 



# Logistic Regression 
reglog = LogisticRegression()
reglog.fit(X_train, y_train)

y_pred = reglog.predict_proba(X_test)
#attention à ne pas calculer le score sur des données modifiées par le SC
print(reglog.score(X_train,y_train))


if __name__ == "__main__":
    main()