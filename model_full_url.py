
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import metrics
import time

#48000 bad queries
#8700 good queries

i = 200

while i < 8700:

    start = time.time()
    #print('started')
    badQueries = open('badQueries_full_url.txt', 'r', encoding='utf-8')
    validQueries = open('goodQueries_full_url.txt', 'r', encoding='utf-8')


    badQueries = list(set(badQueries))
    validQueries = list(set(validQueries))

    num_samples = i
    badQueries = badQueries[:num_samples]
    validQueries = validQueries[:num_samples]

    #print(len(badQueries))
    #print(len(validQueries))
    allQueries = badQueries + validQueries
    yBad = [1 for i in range(0, len(badQueries))]
    yGood = [0 for i in range(0, len(validQueries))]
    y = yBad + yGood
    queries = allQueries


    #print('vectorizing')
    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
    X = vectorizer.fit_transform(queries)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    badCount = len(badQueries)
    validCount = len(validQueries)

    #print('training')
    lgs = LogisticRegression(class_weight={1: 2 * validCount / badCount, 0: 1.0}, solver='lbfgs') # class_weight='balanced')
    lgs.fit(X_train, y_train)

    #print('saving')
    with open('model_full_url.pkl', 'wb') as model_file:
        pickle.dump(lgs, model_file)

    with open('vectorizer_full_url.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    end = time.time()

    predicted = lgs.predict(X_test)

    fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
    auc = metrics.auc(fpr, tpr)

    print("%d" % badCount, end=',')
    print("%d" % validCount, end=',')
    print("%.6f" % (validCount / (validCount + badCount)), end=',')
    print("%f" % lgs.score(X_test, y_test), end=',')  # checking the accuracy
    print("%f" % metrics.precision_score(y_test, predicted), end=',')
    print("%f" % metrics.recall_score(y_test, predicted), end=',')
    print("%f" % metrics.f1_score(y_test, predicted), end=',')
    print("%f" % auc, end=',')
    print("{0}".format(end - start))

    i += 200
