# In charge of admininstrating all
# of the elements in the program

# imports
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, precision_score, recall_score, roc_auc_score
from matplotlib import pyplot
import numpy as np
import itertools
from sklearn.manifold import TSNE
from collections import defaultdict
# scipy test
from scipy.cluster.vq import kmeans, vq, kmeans2
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# personalized import
# from sklearn.cluster import SpectralClustering
from SpectralClustering import SpectralClustering


# class Coordinator
class Coordinator:
    # constructor
    def __init__(self):
        pass

    # join dataSets
    def join(self, pathA, pathB):
        dataSetA = pd.read_csv(pathA, delimiter=',')
        dataSetB = pd.read_csv(pathB, delimiter=',')
        joined = dataSetA.append(
            dataSetB, ignore_index=True)  # continuous idxs
        return joined

    # normalize data set
    def normalize(self, dataSet):
        # clean data
        # dataSet = dataSet.drop('area', axis=1)
        # dataSet = dataSet.drop('kurtosis', axis=1)
        # dataSet = dataSet.drop('perimeter', axis=1)
        dataSet.fillna(method='ffill', inplace=True)

        # change pd dataframe data type to float
        cols = dataSet.columns
        for col in cols:
            dataSet[col] = dataSet[col].astype(float)

        # transpose dataSet (pd frame) and apply technique
        dataSet = dataSet.transpose()

        # Calculate min and max of matrix
        min = 100  # temporal bigger number at first for min
        max = 0
        for _, dim in dataSet.iterrows():  # for dimension or attributes
            tempMin = dim.min()
            tempMax = dim.max()
            if tempMin < min and tempMin != 0:
                min = tempMin
            if tempMax > max:
                max = tempMax

        print('max ', max)
        print('min ', min)

        # apply min max for each value
        idxRow = 0
        for _, dim in dataSet.iterrows():  # for dimension or attributes
            # avoid normalizing first binary column
            for idxCol, value in dim.items():  # for idx or numerosity
                dataSet.iat[int(idxRow), int(idxCol)] = (
                    value-min)/(max-min)  # round(, 20)
            # increment row count
            idxRow += 1

        # to original position
        dataSet = dataSet.transpose()
        # return
        return dataSet

    # Run multiple Spectral Clustering config
    def runConfig(self, completeDf):
        # shuffle
        rowNum, _ = completeDf.shape
        # Transform data to numpy matrix
        completeDf = completeDf.to_numpy().astype(float)
        # Spectral Clustering
        # metricas
        kValues = range(2, 11)  # ks from 2 to 10
        print('kvalues', kValues)
        kMax = max(kValues)
        print(kMax)
        # total mean loss per epoch list
        inertias = []
        # kmeans results
        centroidsList = []
        dists = []
        labelsList = []
        # for all configured kValues
        count = 0
        for k in kValues:
            # Increment counter
            count += 1

            # run ML Model
            clf = SpectralClustering(
                n_clusters=kMax, affinity='rbf', random_state=10)

            # get model predicted labels
            # Training the model and Storing the predicted cluster labels
            # predLabels = clf.fit_predict(completeDf)
            clf.fit(completeDf)
            y_pred = clf.labels_
            inertia = clf.inertia
            clusters = clf.clusters
            spectMap = clf.EigVectMap
            print('spects y total')
            print(spectMap.shape)
            print(completeDf.shape)

            # save inertia
            inertias.append(inertia)

            # scipy test
            km = kmeans(spectMap, k)
            centroide, labels = km
            distTemp = cdist(spectMap, centroide, 'euclidean')
            idx = np.argmin(distTemp, axis=1)
            dist = np.min(distTemp, axis=1)
            avgSS = sum(dist) / spectMap.shape[1]
            print('avgSS', avgSS)

            # append to arrays for graphics
            centroidsList.append(clusters)
            dists.append(avgSS)
            labelsList.append(labels)
        # plot k vs. inertia
        # self.bestKPlot(kValues, inertias)
        self.kDistancePlot(kValues, dists)
        # return
        return labelsList
        # self.kDistancePlot(kValues, labelsList)

    # Run one Spectral Clustering
    def run(self, completeDf, k, labelsList):
        # shuffle
        rowNum, _ = completeDf.shape
        # Transform data to numpy matrix
        completeDfNump = completeDf.to_numpy().astype(float)
        # Spectral Clustering
        # metricas
        kVal = k

        # run ML Model
        clf = SpectralClustering(
            n_clusters=kVal, affinity='nearest_neighbors', random_state=10)  # rbf
        # get model predicted labels
        # Training the model and Storing the predicted cluster labels
        # predLabels = clf.fit_predict(completeDfNump)
        clf.fit(completeDfNump)
        print('completeDfNump', completeDfNump.shape)
        y_pred = clf.labels_
        inertia = clf.inertia
        clusters = clf.clusters
        spectMap = clf.EigVectMap

        # save inertia
        # plot k vs. inertia
        # self.bestKPlot(kValues, inertias)

        # Apply kmeans to spectral map
        print('shape is')
        print(spectMap.shape)
        modelKMeans = KMeans(n_clusters=kVal, random_state=10,
                             init='k-means++').fit(spectMap)
        labelsKmeans = modelKMeans.labels_
        print('ypred is ', y_pred)
        print('labelsmeans is ', labelsKmeans)
        # Generate new matrix
        # spectMapLists = spectMap.transpose().tolist()
        # newMatrix = [[]]*kVal
        # for i in range(spectMap.shape[0]):
        #     row = [row[i] for row in spectMapLists]
        #     if labels[i] == 0:
        #         newMatrix[0].extend(row)
        #     if labels[i] == 1:
        #         newMatrix[1].extend(row)
        #     if labels[i] == 2:
        #         newMatrix[2].extend(row)
        #     if labels[i] == 3:
        #         newMatrix[3].extend(row)
        #     if labels[i] == 4:
        #         newMatrix[4].extend(row)
        #     if labels[i] == 5:
        #         newMatrix[5].extend(row)

        # Plot
        print('labelsKmeans', labelsKmeans)
        print('LABELS ARE HERE', labelsList)
        resultGroups = self.bestKPlot(completeDfNump, labelsKmeans)
        # return
        return resultGroups

    # best k plot with tsne
    def bestKPlot(self, completeDf, labels):
        # Color Mapping
        # Building the label to color pallette
        print('num labels', len(labels))
        print('completeDf.shape')
        print(completeDf.shape)
        colDict = {}
        colDict['blue'] = 'b'
        colDict['yellow'] = 'y'
        colDict['green'] = 'g'
        colDict['red'] = 'r'
        colDict['cyan'] = 'c'
        colDict['magenta'] = 'm'

        colors = list(colDict.values())  # , 'k']

        # get only needed colors for features in completeDf
        _, numCols = completeDf.shape
        colors = colors[:numCols]

        # color chosen pallete
        print('labels', labels)
        cPallette = [colors[label] for label in labels]
        print('cPallette', cPallette)

        # TSNE
        newMatrix = TSNE(early_exaggeration=100, random_state=10).fit_transform(
            completeDf)  # can set rand state
        print('New MATRIX')
        print(newMatrix)
        print(newMatrix.shape)

        # figure
        plt.figure("Best k with tsne Plot")

        # Scatter plot with colors
        x = newMatrix[:, 0]
        y = newMatrix[:, 1]
        print('x', x)
        print('y', y)
        plt.scatter(x, y, c=cPallette)

        # show the plot
        pyplot.show()

        # Place corresponding columns to corresponding clusters
        resultDict = defaultdict(list)
        g0 = []
        g1 = []
        g2 = []
        g3 = []
        g4 = []
        g5 = []
        count = 0
        for lab in labels:
            print('lab', lab)
            if lab == 0:
                g0.append(count)
            if lab == 1:
                g1.append(count)
            if lab == 2:
                g2.append(count)
            if lab == 3:
                g3.append(count)
            if lab == 4:
                g4.append(count)
            if lab == 5:
                g5.append(count)
            # increment count
            count += 1
        # dictionary result to print in text file
        colorNames = list(colDict.keys())
        resultDict[colorNames[0]].extend(g0)
        resultDict[colorNames[1]].extend(g1)
        resultDict[colorNames[2]].extend(g2)
        resultDict[colorNames[3]].extend(g3)
        resultDict[colorNames[4]].extend(g4)
        resultDict[colorNames[5]].extend(g5)

        # return
        return resultDict

        # plot leyend
        # labcolors = []
        # labNames = []
        # count = 0
        # for colorSelect in colors:
        #     count += 1
        #     clr = plt.scatter(x, y, color=colorSelect)
        #     labcolors.append(clr)
        #     labNames.append('Class ' + str(count))
        # plt.legend(tuple(labcolors), tuple(labNames))

    # k vs inertia graphic
    def kDistancePlot(self, kValues, distances):
        pyplot.plot(kValues, distances, marker='.',
                    label='K vs. D')
        pyplot.xlabel('K Values')
        pyplot.ylabel('Distance')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

    # function to call graphics

    def callGraphics(self, fprs, tprs, precs, recs, epochs, meanLossList):
        self.AUCplot(fprs, tprs)
        self.presitionRecallPlot(precs, recs)
        self.lossEpochsPlot(epochs, meanLossList)

    # Run AUC graph
    def AUCplot(self, fprMatrix, tprMatrix):
        Fpr = []
        Tpr = []
        minFpr = 100  # initial val
        minTpr = 100  # initial val
        # get min size element
        for elem in fprMatrix:
            # print('elem fpr ', elem)
            # print('shape fpr ', elem.shape)
            elmSize = len(elem)
            # print('fpr elm size ', elmSize)
            if elmSize < minFpr:
                minFpr = elmSize
        for elem in tprMatrix:
            # print('elem tpr ', elem)
            # print('shape tpr ', elem.shape)
            elmSize = len(elem)
            # print('tpr elm size ', elmSize)
            if elmSize < minTpr:
                minTpr = elmSize

        # make same length to allow sum
        for elem in fprMatrix:
            elmSize = len(elem)
            # print(elmSize)
            Fpr.append(elem[:minFpr])
        for elem in tprMatrix:
            elmSize = len(elem)
            # print(elmSize)
            Tpr.append(elem[:minTpr])

        # make all numpy arrays the same
        meanFpr = sum(Fpr) / len(Fpr)
        meanTpr = sum(Tpr) / len(Tpr)
        # print('AUC means')
        # print(meanFpr, meanTpr)
        pyplot.plot(meanFpr, meanTpr, marker='.',
                    label='Area Under the Curve')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

    # Run precission recall graph
    def presitionRecallPlot(self, precMatrix, recallMatrix):
        Prec = []
        Rec = []
        minPrec = 100  # initial val
        minRec = 100  # initial val
        # get min size element
        for elem in precMatrix:
            # print('elem prec ', elem)
            # print('shape prec ', elem.shape)
            elmSize = len(elem)
            # print('prec elm size ', elmSize)
            if elmSize < minPrec:
                minPrec = elmSize
        for elem in recallMatrix:
            # print('elem recall ', elem)
            # print('shape recall ', elem.shape)
            elmSize = len(elem)
            # print('prec recall size ', elmSize)
            if elmSize < minRec:
                minRec = elmSize
        # make same length to allow sum
        for elem in precMatrix:
            Prec.append(elem[:minPrec])
        for elem in recallMatrix:
            Rec.append(elem[:minRec])
        # make all numpy arrays the same
        meanPrec = sum(Prec) / len(Prec)
        meanRecall = sum(Rec) / len(Rec)
        # print('Prec Rec means')
        # print(meanPrec, meanRecall)
        # print('\n Precision vs Recall: ')
        # precision, recall, _ = precision_recall_curve(precMatrix, recallMatrix)
        pyplot.plot(meanRecall, meanPrec, marker='.',
                    label='Precision vs. Recall')
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

    # Run mean loss vs epochs
    def lossEpochsPlot(self, epochs, meanLossList):
        pyplot.plot(epochs, meanLossList, marker='.',
                    label='Mean Loss vs. Epochs')
        pyplot.xlabel('Mean Loss')
        pyplot.ylabel('Epochs')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
