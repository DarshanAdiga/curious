import re
import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

class_dict = {
        'display': 0,
        'unknown': 1,
        'camera': 2,
        'dimension': 3,
        'hardware-keys': 4,
        'sim-card': 5,
        'rear-camera': 6,
        'accessories': 7,
        'features': 8,
        'secondary-storage': 9,
        'OS': 10,
        'battery': 11,
        'front-camera': 12,
        'CPUs': 13,
        'primary-memory': 14,
        'price': 15,
        'expandable-memory': 16,
        'hardware-quality': 17,
        'applications': 18,
    }


def to_ids(train_classes):
    res_classes = []
    for c in train_classes:
        res_classes.append(class_dict[c])
    return res_classes


def get_class(category):
    for ind, cat in class_dict.items():
        if cat == category:
            return ind
    return -1


def clean_sent(sent):
    letters_only = re.sub("[^a-zA-Z0-9]", " ", sent)
    lower_case = letters_only.lower()
    return lower_case


def trainer():
    base = "res/"
    train = pandas.read_csv(base + 'train.csv', header=0, quoting=2)
    # print(train.columns.values)
    # print(train["sentence"][0])

    clean_train_sent = []
    for s in train["sentence"]:
        clean_train_sent.append(clean_sent(s))

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                 preprocessor=None, stop_words="english", max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_train_sent).toarray()
    train_classes = to_ids(train["class"])
    # print(train_data_features[0])
    print("Training feature size:" + str(train_data_features.shape))

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train_classes)

    test = ["The rear 23-megapixel camera does a wonderful job snapping pictures",
            "5.5-inch screen with 1,920x1080-pixel resolution"]
    for t in test:
        ct = clean_sent(t)
        ct_features = vectorizer.transform([ct]).toarray()
        res = forest.predict(ct_features)
        print(t + ":>>>>" + get_class(res[0]))


if __name__ == '__main__':
    trainer()