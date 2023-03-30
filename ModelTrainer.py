import pandas as pd
from sklearn import metrics
from sklearn import neighbors
import os
import joblib
class ModelTrainer:
    def __init__(self):
        train_file = pd.read_csv('train.csv').drop(columns=['datasetId'])
        test_file = pd.read_csv('test.csv').drop(columns=['datasetId'])
        X_train = train_file.drop(columns=['condition'])
        y_train = train_file['condition']
        X_test = test_file.drop(columns=['condition'])
        y_test = test_file['condition']

        classifier = neighbors.KNeighborsClassifier()
        print("Start Training...")
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, predicted))
        print("Precision:", metrics.precision_score(y_test, predicted, average='micro'))
        print("Recall:", metrics.recall_score(y_test, predicted, average='micro'))
        print("F1 Score:", metrics.f1_score(y_test, predicted, average='micro'))

        path = os.path.join(os.path.dirname(__file__), 'model.joblib')
        joblib.dump(classifier, path)

if __name__ == '__main__':
    trainer = ModelTrainer()
