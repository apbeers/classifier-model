from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import random

bad_queries_tld = list(set(open('badqueries_tld.txt', 'r', encoding='utf-8')))
good_queries_tld = list(set(open('goodqueries_tld.txt', 'r', encoding='utf-8')))

bad_queries_full_url = list(set(open('badqueries_full_url.txt', 'r', encoding='utf-8')))
good_queries_full_url = list(set(open('goodqueries_full_url.txt', 'r', encoding='utf-8')))

# Will take 2% of each dataset to use for testing

print('Assigning Categories')
tld_x = bad_queries_tld + good_queries_tld
tld_y_bad = [1 for i in range(0, len(bad_queries_tld))]
tld_y_good = [0 for i in range(0, len(good_queries_tld))]
tld_y = tld_y_bad + tld_y_good

full_url_x = good_queries_full_url + good_queries_full_url
full_url_y_bad = [1 for i in range(0, len(bad_queries_full_url))]
full_url_y_good = [0 for i in range(0, len(good_queries_full_url))]
full_url_y = full_url_y_bad + full_url_y_good

print('Vectorizing')
tld_vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
tld_vector = tld_vectorizer.fit_transform(tld_x)
tld_x_train, tld_x_test, tld_y_train, tld_y_test = train_test_split(tld_vector, tld_y)

full_url_vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
full_url_vector = full_url_vectorizer.fit_transform(full_url_x)
full_x_bad_train, full_x_bad_test, full_y_bad_train, full_y_bad_test = train_test_split(full_url_vector, full_url_y)


print('Training')
tld_lgs = LogisticRegression(class_weight={1: 2 * len(bad_queries_tld_train) / len(good_queries_tld_train), 0: 1.0}, solver='lbfgs') # class_weight='balanced')
tld_lgs.fit(tld_vector, tld_y)


full_url_lgs = LogisticRegression(class_weight={1: 2 * len(bad_queries_full_url_train) / len(good_queries_full_url_train), 0: 1.0}, solver='lbfgs') # class_weight='balanced')
tld_lgs.fit(full_url_vector, full_url_y)

print('Validating')
for query in bad_queries_tld_test:
    tld_probability = tld_lgs.predict_proba(tld_vectorizer.transform([query]))
    tld_percent_true = float(tld_probability[0][0])
    tld_percent_false = float(tld_probability[0][1])

    tld_accuracy = tld_percent_true - tld_percent_false
    if tld_accuracy < 0:
        tld_accuracy *= -1

    full_url_probability = full_url_lgs.predict_proba(full_url_vectorizer.transform([query]))
    full_url_percent_true = float(full_url_probability[0][0])
    full_url_percent_false = float(full_url_probability[0][1])

    full_url_accuracy = full_url_percent_true - full_url_percent_false
    if full_url_accuracy < 0:
        full_url_accuracy *= -1

    if tld_accuracy > full_url_accuracy:
        prediction = tld_lgs.predict(tld_vectorizer.transform([query]))
    else:
        prediction = full_url_lgs.predict(full_url_vectorizer.transform([query]))

    if prediction:
        print('correct')
    else:
        print('wrong')
