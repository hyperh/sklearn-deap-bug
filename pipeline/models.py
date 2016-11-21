from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def getModels(stepName):
    prefix = stepName + '__'
    return [
        {
            stepName: [DummyClassifier()],
            prefix + 'strategy': ['stratified', 'most_frequent'],
        },
        {
            stepName: [SVC()],
            prefix + 'C': [10 ** i for i in range(0, 4)],
            prefix + 'gamma': [10 ** -i for i in range(0, 7)],
            prefix + 'kernel': ['rbf'],
        },
        {
            stepName: [RandomForestClassifier()],
            prefix + 'n_estimators': [5, 10, 20, 40],
        },
        {
            stepName: [KNeighborsClassifier()],
            prefix + 'n_neighbors': [2, 4, 6, 8, 10, 15, 20],
            prefix + 'weights': ['uniform', 'distance']
        },
        {
            stepName: [MLPClassifier(max_iter=1000)],
            prefix + 'hidden_layer_sizes': [
                (100,),
                (100, 50, 25),
                (50, 25, 10),
                (5, 5, 5),
                (50,)
            ],
            prefix + 'activation': [
                'logistic', 'tanh', 'relu'
            ],
            prefix + 'alpha': [
                10 ** -4, 10 ** -3, 10 ** -2
            ]
        },
    ]
