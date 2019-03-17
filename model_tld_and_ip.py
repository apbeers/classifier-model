
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

print('started')
badQueries = open('badQueries.txt', 'r', encoding='utf-8')
validQueries = open('goodQueries.txt', 'r', encoding='utf-8')

badQueries = list(set(badQueries))
validQueries = list(set(validQueries))
print(len(badQueries))
print(len(validQueries))
allQueries = badQueries + validQueries
yBad = [1 for i in range(0, len(badQueries))]
yGood = [0 for i in range(0, len(validQueries))]
y = yBad + yGood
queries = allQueries


print('vectorizing')
vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
X = vectorizer.fit_transform(queries)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

badCount = len(badQueries)
validCount = len(validQueries)

print('training')
lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0}, solver='lbfgs') # class_weight='balanced')
lgs.fit(X_train, y_train)

print('saving')
with open('model_tld.pkl', 'wb') as model_file:
    pickle.dump(lgs, model_file)

with open('vectorizer_tld.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
