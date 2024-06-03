#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar, DefaultDict
from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Problem 1: binary classification
############################################################

############################################################
# Problem 1a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # ### START CODE HERE ###
    words = x.split()
    feature_vec = {}
    for word in words:
        feature_vec[word] = feature_vec.get(word, 0) + 1
    return feature_vec
    # ### END CODE HERE ###


############################################################
# Problem 1b: stochastic gradient descent

T = TypeVar("T")


def learnPredictor(
        trainExamples: List[Tuple[T, int]],
        validationExamples: List[Tuple[T, int]],
        featureExtractor: Callable[[T], FeatureVector],
        numEpochs: int,
        eta: float,
) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    """
    weights = {}  # feature => weight

    # ### START CODE HERE ###
    def predict(x):
        phi = featureExtractor(x)
        if dotProduct(phi, weights) >= 0:
            return 1
        else:
            return -1

    for epoch in range(numEpochs):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            if dotProduct(weights, phi) * y < 1:
                increment(weights, eta * y, phi)
        print(f"Epoch:{epoch + 1}, Training error: {evaluatePredictor(trainExamples, predict)},"
              f"Test error:{evaluatePredictor(validationExamples, predict)}")
    # ### END CODE HERE ###
    return weights


############################################################
# Problem 1c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        phi = None
        y = None
        # ### START CODE HERE ###
        feature_keys = random.sample(weights.keys(), random.randint(1, len(weights)))

        phi = {}
        for key in feature_keys:
            phi[key] = random.randint(1, 10)

        y = 1 if dotProduct(weights, phi) >= 0 else -1

        # ### END CODE HERE ###
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 1d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x):
        # ### START CODE HERE ###
        x = x.replace(" ", "")
        feature_vector = {}

        for i in range(len(x) - n + 1):
            ngram = x[i:i + n]
            feature_vector[ngram] = feature_vector.get(ngram, 0) + 1

        return feature_vector
        # ### END CODE HERE ###

    return extract


############################################################
# Problem 1e:
#
# Helper function to test 1e.
#
# To run this function, run the command from termial with `n` replaced
#
# $ python -c "from submission import *; testValuesOfN(n)"
#


def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be submitted.
    """
    trainExamples = readExamples("polarity.train")
    validationExamples = readExamples("polarity.dev")
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(
        trainExamples, validationExamples, featureExtractor, numEpochs=20, eta=0.01
    )
    outputWeights(weights, "weights")
    outputErrorAnalysis(
        validationExamples, featureExtractor, weights, "error-analysis"
    )  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    validationError = evaluatePredictor(
        validationExamples,
        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1),
    )
    print(
        (
                "Official: train error = %s, validation error = %s"
                % (trainError, validationError)
        )
    )


############################################################
# Problem 2b: K-means
############################################################


def kmeans(
        examples: List[Dict[str, float]], K: int, maxEpochs: int
) -> Tuple[List, List, float]:
    """
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    """
    # ### START CODE HERE ###
    centroids = random.sample(examples, K)
    num_examples = len(examples)

    for epoch in range(maxEpochs):
        assignments = [min(range(K), key=lambda j: sum((examples[i][key] - centroids[j].get(key, 0)) ** 2
                                                       for key in examples[i])) for i in range(num_examples)]
        new_centroids = [{} for _ in range(K)]
        cluster_sizes = [0] * K

        for i in range(num_examples):
            cluster = assignments[i]
            cluster_sizes[cluster] += 1
            for key, value in examples[i].items():
                new_centroids[cluster][key] = new_centroids[cluster].get(key, 0) + value

        for j in range(K):
            if cluster_sizes[j] > 0:
                for key in new_centroids[j]:
                    new_centroids[j][key] /= cluster_sizes[j]

        # Check for convergence
        if new_centroids == centroids:
            break

        centroids = new_centroids

        # Calculate the final reconstruction loss
    loss = sum(sum((examples[i][key] - centroids[assignments[i]].get(key, 0)) ** 2 for key in examples[i])
               for i in range(num_examples))

    return centroids, assignments, loss
    # ### END CODE HERE ###
