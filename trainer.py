import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import svm

with open("res/sample_train.csv", 'r') as train_file:
    reader = csv.reader(train_file)
    print(next(reader))
    cats = []
    comments = []
    for line in reader:
        cats.append(line[0])
        comments.append(line[2])
        print(line[0], ':', line[2])

    # count_vect = CountVectorizer()
    # train_counts = count_vect.fit_transform(comments)
    # tfidf_transformer = TfidfTransformer()
    # train_tfidf = tfidf_transformer.fit_transform(train_counts)
    # print(train_tfidf[0, 0])

    clf = svm.SVC()
    print(comments)
    clf.fit(np.array(cats), np.array(comments))
