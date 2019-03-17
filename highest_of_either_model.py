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


test_percent = .2

print('Shuffling')
np.random.shuffle(bad_queries_tld)
np.random.shuffle(good_queries_tld)
np.random.shuffle(bad_queries_full_url)
np.random.shuffle(good_queries_full_url)



print('Splitting')
bad_queries_tld_train, bad_queries_tld_test = bad_queries_tld[int(len(bad_queries_tld) * (1-test_percent)):], bad_queries_tld[:int(len(bad_queries_tld) - (len(bad_queries_tld) * test_percent))]
good_queries_tld_train, good_queries_tld_test = good_queries_tld[int(len(good_queries_tld) * (1-test_percent)):], bad_queries_tld[:int(len(good_queries_tld) - (len(good_queries_tld) * test_percent))]

bad_queries_full_url_train, bad_queries_full_url_test = bad_queries_full_url[int(len(bad_queries_full_url) * (1-test_percent)):], bad_queries_full_url[:int(len(bad_queries_full_url) - (len(bad_queries_full_url) * test_percent))]
good_queries_full_url_train, good_queries_full_url_test = good_queries_full_url[int(len(good_queries_full_url) * (1-test_percent)):], good_queries_full_url[:int(len(good_queries_full_url) - (len(good_queries_full_url) * test_percent))]

print('Assigning Categories')
tld_all_train = bad_queries_tld_train + good_queries_tld_train
tld_y_bad = [1 for i in range(0, len(bad_queries_tld_train))]
tld_y_good = [0 for i in range(0, len(good_queries_tld_train))]
tld_y = tld_y_bad + tld_y_good

full_url_all_train = bad_queries_full_url_train + good_queries_full_url_train
full_url_y_bad = [1 for i in range(0, len(bad_queries_full_url_train))]
full_url_y_good = [0 for i in range(0, len(good_queries_full_url_train))]
full_url_y = full_url_y_bad + full_url_y_good

print('Vectorizing')
tld_vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
tld_vector = tld_vectorizer.fit_transform(tld_all_train)

full_url_vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(1, 3))
full_url_vector = full_url_vectorizer.fit_transform(full_url_all_train)

print('Training')
tld_lgs = LogisticRegression(class_weight={1: 2 * len(bad_queries_tld_train) / len(good_queries_tld_train), 0: 1.0}, solver='lbfgs') # class_weight='balanced')
tld_lgs.fit(tld_vector, tld_y)

full_url_lgs = LogisticRegression(class_weight={1: 2 * len(bad_queries_full_url_train) / len(good_queries_full_url_train), 0: 1.0}, solver='lbfgs') # class_weight='balanced')
full_url_lgs.fit(full_url_vector, full_url_y)

print('Validating')
correct = 0
incorrect = 0
tld_model = 0
full_url_model = 0
for query in bad_queries_tld:
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
        tld_model += 1
    else:
        prediction = full_url_lgs.predict(full_url_vectorizer.transform([query]))
        full_url_model += 1

    if prediction:
        correct += 1
    else:
        incorrect += 1


print('correct: {0}'.format(correct))
print('incorrect: {0}'.format(incorrect))
print('full url model: {0}'.format(full_url_model))
print('tld model: {0}'.format(tld_model))
print('accuracy: {0}'.format(correct/(correct + incorrect)))
print('num samples: {0}'.format(correct + incorrect))

print('{0}'.format(correct + incorrect), end=',')
print('{0}'.format(correct/(correct + incorrect)), end=',')
print('{0}'.format(full_url_model/(full_url_model + tld_model)), end='')
