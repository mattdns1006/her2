import pandas as pd
import pdb
import numpy as np
import sys


if __name__ == "__main__":
    path = "data/"
    groundTruth = pd.read_csv(path + "groundTruth.csv")
    nObs = groundTruth.shape[0]
    try:
        splitPerc = float(sys.argv[1])
    except IndexError:
        print("Please input split percentage.")

    splitMark = np.floor(splitPerc*nObs).astype("int8")

    randomPerm = np.random.permutation(groundTruth.shape[0])
    trainIdx, testIdx = randomPerm[:splitMark], randomPerm[splitMark:]
    intersection = np.intersect1d(trainIdx,testIdx)
    assert np.size(intersection) == 0, "Train, test sets share some values"

    train, test = groundTruth.loc[trainIdx], groundTruth.loc[testIdx]
    print("Number of observations in train/test %d/%d" % (train.shape[0],test.shape[0]))
    train.to_csv(path + "groundTruthTrain.csv",index=0)
    test.to_csv(path + "groundTruthTest.csv",index=0)

    print("Value counts of train/test ==>")
    print([x.ix[:,1].value_counts() for x in [train,test]])
	



