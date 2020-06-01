# In The Name Of Allah

# Name: Naive Bayes
# Purpose: Implementation Of Naive Bayes, Classification Algorithm
# Programmer: Ali Salimi
# Date: 1396/10/08


# Imports
from collections import defaultdict
from random import shuffle


# NaiveBayes Class Definition
class NaiveBayes:

    # for creating model, learning
    @staticmethod
    def learn(ds):
        dsLen = len(ds)
        lp = defaultdict(float)
        fp = []
        for i in range(0, 6):
            fp.append(defaultdict(lambda: defaultdict(float)))

        for sample in ds:
            i = 0
            lp[sample[-1]] += 1
            for item in sample[:-1]:
                fp[i][item][sample[-1]] += 1
                i += 1

        i = 0
        for feature in fp:
            for key, value in feature.items():
                for key2, value2 in value.items():
                    fp[i][key][key2] /= lp[key2]
            i += 1

        for key, value in lp.items():
            lp[key] /= dsLen

        return (lp, fp)

    # for predicting label of a sample, predicting
    @staticmethod
    def predict(p, sample):
        LP, FP = p

        LsP = defaultdict(float)

        for key, value in LP.items():
            i = 0
            LsP[key] = value
            for item in sample:
                LsP[key] *= FP[i][item][key]
                i += 1

        key, value = max(LsP.items(), key=lambda a: a[1])
        return key

    # for validating this algorithm on a data set
    @staticmethod
    def crossValidation(ds):
        shuffle(ds)
        dsLen = len(ds)
        partLen = int(dsLen / 10)

        percents = []
        for i in range(0, 10):
            predictPart = ds[(i * partLen) : ((i + 1) * partLen)]
            learnPart = ds[:(i * partLen)] + ds[((i + 1) * partLen):]

            possibilitys = NaiveBayes.learn(learnPart)
            percent = 0.0
            for sample in predictPart:
                if NaiveBayes.predict(possibilitys, sample[:-1]) == sample[-1]:
                    percent += 1
            percent /= partLen
            percents.append(percent)
        avrage = sum(percents) / float(len(percents))

        return (avrage, percents)


# Reading Data Set From File
DSLocation = "krkopt.data"
DS = []
for line in open(DSLocation):
    line = line.replace("\n", "")
    DS.append(line.split(','))

# Invoking Class Functions On This Data Set
lp, fp = NaiveBayes.learn(DS)
print("Labels Possibility:", lp)
print("Features Possibility:", fp)
print("For ['b', '1', 'c', '1', 'd', '1'] this label is predicted:",
      NaiveBayes.predict((lp, fp), ['b', '1', 'c', '1', 'd', '1']))
print('Cross Validation:', NaiveBayes.crossValidation(DS))
