import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


#
dataset = pd.read_csv(“/Users/benothmane/Desktop/Projet IA/fake_or_real_news.csv”)
dataset.head()
#
print(dataset.shape)
# We have 6335 rows and 4 columns.
#
print(dataset[‘title’].isnull())
print(dataset[‘text’].isnull())
print(dataset[‘label’].isnull())
#
  list(dataset.columns)

#
dataset.drop([‘Unnamed: 0’], axis=1, inplace=True)
#We have a cleaner data. 
dataset.head()  
#
x = np.array(dataset[“title”])
y = np.array(dataset[“label”])
cv = CountVectorizer()
x = cv.fit_transform(x)
#
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
#
snews_headline = “Ukraine: Austrian leader, Putin meet…other new developments”
data = cv.transform([news_headline]).toarray()
print(model.predict(data))
#
news_headline = “A lion was found flying in South America”
data = cv.transform([news_headline]).toarray()
print(model.predict(data))
#          