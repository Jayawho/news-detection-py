import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#naviebayes,linear regression,linearSVM
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



df_real=pd.read_csv('True.csv')
df_fake=pd.read_csv('Fake.csv')


df_real['label']='REAL'
df_fake['label']='FAKE'


df=pd.concat([df_real, df_fake], ignore_index=True)


df=df.dropna(subset=['text'])


df=df[['text', 'label']]


df=df.sample(frac=1, random_state=42).reset_index(drop=True)


df.to_csv('combined_news.csv', index=False)




stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()


df['cleantext']=df['text'].astype(str).str.lower()


df['cleantext']=df['cleantext'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

df['cleantext']=df['cleantext'].apply(lambda x: re.sub(r'\d+', '', x))


df['cleantext']=df['cleantext'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in stop_words]))



df.to_csv('preprocessed_news.csv', index=False)


X=df['text']  
y=df['label']


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)


tfidf=TfidfVectorizer(stop_words='english', max_df=0.2)
X_train_tfidf= tfidf.fit_transform(X_train)
X_test_tfidf=tfidf.transform(X_test)

logclf= LogisticRegression(max_iter=1000)
nbclf=   MultinomialNB()
svmclf= LinearSVC()


voting_clf= VotingClassifier(estimators=[('lr', logclf),('nb', nbclf),('svm', svmclf)],voting='hard'  )#soft voting did not work :(

voting_clf.fit(X_train_tfidf,y_train)

y_pred =voting_clf.predict(X_test_tfidf)

print("VotingClassifier Accuracy:",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred,zero_division=1))





from sklearn.pipeline import Pipeline
voting_model =Pipeline([('tfidf', TfidfVectorizer()),('voting', voting_clf)])



import joblib


joblib.dump(voting_model,'fake_news_model.pkl')
print("Model saved as 'fake_news_model.pkl'")



