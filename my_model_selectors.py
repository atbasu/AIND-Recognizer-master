import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        sBics = []
        for nStates in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm = self.base_model(nStates)
                lL = hmm.score(self.X, self.lengths)
                nDataPts, f = self.X.shape
                nFreeParams = ( nStates ** 2 ) + ( 2 * nStates * f ) - 1
                sBic = -2 * lL + nFreeParams * np.log(nDataPts)
                sBics.append(tuple([sBic, hmm]))
            except:
                pass
        return min(sBics, key = lambda x: x[0])[1] if sBics else None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def calcLogLikelihoodOtherWords(self, model, oWords):
        return [model[1].score(word[0], word[1]) for word in oWords]


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        oWords = []
        models = []
        sDics = []
        for word in self.words:
            if word != self.this_word:
                oWords.append(self.hwords[word])
        try:
            for nStates in range(self.min_n_components, self.max_n_components + 1):
                hmm = self.base_model(nStates)
                lLOrigWord = hmm.score(self.X, self.lengths)
                models.append((lLOrigWord, hmm))

        
        except Exception as e:
            # catches situations where there are more parameters to fit than there are samples
            # resulting in invalid model
            pass
        for index, model in enumerate(models):
            lLOrigWord, hmm = model
            sDic = lLOrigWord - np.mean(self.calcLogLikelihoodOtherWords(model, oWords))
            sDics.append(tuple([sDic, model[1]]))
        return max(sDics, key = lambda x: x[0])[1] if sDics else None


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        kf = KFold(n_splits = 3, shuffle = False, random_state = None)
        lLs = []
        sCVs = []


        for nStates in range(self.min_n_components, self.max_n_components + 1):
            try:
                if len(self.sequences) > 2:
                    for trIndex, teIndex in kf.split(self.sequences):
                        
                        self.X, self.lengths = combine_sequences(trIndex, self.sequences)

                        xTest, lTest = combine_sequences(teIndex, self.sequences)

                        hmm = self.base_model(nStates)
                        lL = hmm.score(xTest , lTest)
                else:
                    hmm = self.base_model(nStates)
                    lL = hmm.score(self.X, self.lengths)

                lLs.append(lL)

                sCVsAvg = np.mean(lLs)
                sCVs.append(tuple([sCVsAvg, hmm]))

            except Exception as e:
                pass
        return max(sCVs, key = lambda x: x[0])[1] if sCVs else None
