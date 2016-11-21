from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
import math
from sklearn.decomposition import PCA


def getDimTransformers(stepName, numFeatures):
    prefix = stepName + '__'

    maxExp2 = math.ceil(math.log(numFeatures, 2)) + 1
    numFeats = [
        2 ** i if i < maxExp2 - 1 else
        numFeatures
        for i in range(1, maxExp2)
    ]

    maxVal = min(numFeatures, 32)
    maxExp2 = math.ceil(math.log(maxVal, 2)) + 1
    numComponents = [
        2 ** i if i < maxExp2 - 1 else
        numFeatures
        for i in range(1, maxExp2)
    ]

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)
    return [
        # {
        #     stepName: [None],
        # },
        {
            stepName: [SelectFromModel(lsvc)]
        },
        {
            stepName: [SelectKBest(score_func=f_classif)],
            prefix + 'k': numFeats,
        },
        {
            stepName: [PCA()],
            prefix + 'n_components': numComponents,
        },
    ]
