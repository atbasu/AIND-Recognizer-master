import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for wordId in range(0, len(test_set.get_all_Xlengths())):
        wordFeatListsSequences, sequencesLength = test_set.get_item_Xlengths(wordId)
        wordLLs = {}

        for word, model in models.items():
            try:
                score = model.score(wordFeatListsSequences, sequencesLength)
                wordLLs[word] = score
            except:
                wordLLs[word] = float("-inf")
                continue
        # Probabilities appended with probability list
        probabilities.append(wordLLs)
        # Guesses appended with calculation of word with maximum score (log likelihood) for each model
        guesses.append(max(wordLLs, key = wordLLs.get))

    return probabilities, guesses
