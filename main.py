import runner
from functools import partial
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.datasets import make_classification


GA = partial(
    EvolutionaryAlgorithmSearchCV,
    verbose=1,
    population_size=50,
    gene_mutation_prob=0.10,
    gene_crossover_prob=0.5,
    tournament_size=3,
    generations_number=5,
    n_jobs=4,
    scoring='accuracy'
)

X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False
)

runner.run(
    features=X,
    labels=y,
    hyperOpt=GA,
)
