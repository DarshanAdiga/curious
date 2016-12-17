import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import svm

with open("res/train.csv", 'r') as train_file:
    reader = csv.reader(train_file)
    #print(next(reader))
    cats = []
    comments = []
    for line in reader:
        cats.append(line[0])
        comments.append(line[2])
        #print(line[0], ':', line[2])

    #count_vect = CountVectorizer()
    #train_counts = count_vect.fit_transform(comments)
    # tfidf_transformer = TfidfTransformer()
    # train_tfidf = tfidf_transformer.fit_transform(train_counts)
    # print(train_tfidf[0, 0])

    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    examples = ['I will kill you']

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', MultinomialNB())])
    pipeline.fit(comments, cats)
    predictions = pipeline.predict(examples)

    # print(train_counts)
    #classifier = MultinomialNB()
    #classifier.fit(train_counts, cats)

    #example_counts = count_vect.transform(examples)
    #predictions = classifier.predict(example_counts)
    #print(predictions)
    if '1' in predictions:
        print(examples)
        print("Abusive")
    else:
        print(examples)
        print("NonAbusive")