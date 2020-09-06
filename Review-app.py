%%writefile app-ml.py
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def get_data():
  value = st.text_input("")
  data = {'Text':value}
  return data

def user_input():
  features = pd.DataFrame(get_data(), index= [0])
  return features

st.title("Machine Learning Model")
st.subheader("I have created an ML model using Python")
st.write("ML is awesome")

df1 = user_input()
st.write(df1)

df = pd.read_csv('Reviews.csv', encoding='utf-8',nrows= 6000)
def sentiment_rating(rating):
    # Replacing ratings of 1,2,3 with 0 (not good) and 4,5 with 1 (good)
    if(int(rating) == 1 or int(rating) == 2 or int(rating) == 3):
        return 0
    else: 
        return 1
df.Score= df.Score.apply(sentiment_rating) 
# df.Score.head()
df.Score.value_counts().plot.pie(autopct='%1.1f%%')

x = df.iloc[:,9].values
y = df.iloc[:,6].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.7,random_state=100)


# Pipeline([('Variable 1',Method 1()),('Variable 2',Method 2())])
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',MultinomialNB())])
text_model.fit(x_train,y_train)
y_pred = text_model.predict(df1)
st.write(y_pred)
