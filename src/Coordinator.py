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
import sklearn.neural_network as nn
import numpy as np
import itertools


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

        # apply min max for each value
        idxRow = 0
        for _, dim in dataSet.iterrows():  # for dimension or attributes
            # avoid normalizing first binary column
            for idxCol, value in dim.items():  # for idx or numerosity
                dataSet.iat[int(idxRow), int(idxCol)] = round(
                    (value-min)/(max-min), 6)
            # increment row count
            idxRow += 1

        # to original position
        dataSet = dataSet.transpose()
        # return
        return dataSet

    # Run one Neural network configurations
    def runConfig(self, completeDf, k, hidLayers, activFunc):
        # shuffle
        rowNum, _ = completeDf.shape
        completeDf = completeDf.sample(rowNum, random_state=1)
        # Target column and other columns of DataSet
        targ = completeDf.iloc[:, 0].to_numpy().astype(int)
        features = completeDf.iloc[:, 1:].to_numpy()
        # MLPClassifier configuracion
        # topologia
        hiddenLayers = hidLayers
        activFunc = activFunc
        # hiper parametros
        lrs = [0.01, 0.05, 0.1, 0.3, 0.5]
        # epochs = [500, 600, 700, 800, 900, 1000]
        epochs = [5, 10, 15, 20, 25, 30]
        # total mean loss per epoch list
        meanLossList = []
        higherAuc = 0
        # loss classified for distinct epochs
        meanLoss1 = []
        meanLoss2 = []
        meanLoss3 = []
        meanLoss4 = []
        meanLoss5 = []
        meanLoss6 = []
        # generate result dictionary
        dictNN = {}
        # configurations
        dictNN['Configuraciones:'] = 'Topología'
        dictNN['capas Ocultas:'] = str(hiddenLayers)
        dictNN['función de Activación:'] = activFunc
        dictNN['configuraciones:'] = 'Hiperparámetros'
        # , shuffle=True)
        # for epoch and learning rate combinations
        cartesian = itertools.product(epochs, lrs)
        count = 0
        for epoch, lr in cartesian:
            # Increment counter
            count += 1
            # desempeño
            meanAccNN = []
            meanPrecNN = []
            meanRecallNN = []
            meanRocAucNN = []
            meanLossNN = []  # loss for every case

            # generate kfold splits
            kFold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
            splitsIter = kFold.split(features, targ)
            # run neural network calculations through all folds
            print(f'iteration {count}\n')
            for trainIndex, testIndex in splitsIter:
                # train and test of every fold
                X_train, X_test = features[trainIndex], features[testIndex]
                y_train, y_test = targ[trainIndex], targ[testIndex]
                # Classifier object
                clf = nn.MLPClassifier(
                    # verbose=True,
                    # batch_size=200,
                    hidden_layer_sizes=hiddenLayers,
                    # activation func default is relu
                    activation=activFunc,
                    early_stopping=False,
                    solver='sgd',
                    # learning_rate='constant',  # no change as epochs advance
                    learning_rate_init=lr,
                    # max_iter=epoch,
                    random_state=1,  # To change random seed
                    # tol=0.001  # minimum improvement in cost (tol)
                    # number of epochs it was less than tol, to halt the process
                    # n_iter_no_change=1000
                )
                # fit fold data set
                clf.fit(X_train, y_train)
                lossVals = clf.loss_curve_
                # binary prediction
                y_pred = clf.predict(X_test)
                # probabilistic prediction
                y_prob = clf.predict_proba(X_test)
                # print('ypred, ', y_pred)
                # print('ytest, ', y_test)
                # print('yprob', y_prob)
                # DESEMPEÑO
                meanAccNN.append(accuracy_score(y_test, y_pred))
                meanPrecNN.append(precision_score(y_test, y_pred))
                meanRecallNN.append(recall_score(y_test, y_pred))
                meanRocAucNN.append(roc_auc_score(y_test, y_prob[:, 1]))
                meanLossNN.append(sum(
                    lossVals) / len(lossVals))
                # meanLossNN.append(clf.loss_)
                # print(accuracy_score(y_test, y_pred))
                # print(precision_score(y_test, y_pred))
                # print(recall_score(y_test, y_pred))
                # print(roc_auc_score(y_test, y_prob[:, 1]))
                print(clf.loss_)
                # accuracy_score(y_test, y_pred)
                # print('acc ', roc_auc_score(y_test, y_prob[:, 0]))
                # print('prec ', precision_score(y_test, y_pred))
                # print('roc auc score ', roc_auc_score(y_test, y_prob[:, 1]))
                # print(clf.classes_)
            # ESCRIBIR RESULTADOS por cada epoch y lr
            # DESEMPEÑO
            # metrics to average
            if(meanAccNN):  # avoid case when folds end
                # print('im in!')
                dictNN[f'\nEPOCH and Lr {count}:'] = str(epoch) + '/' + str(lr)
                dictNN[f'accuracy promedio {count}:'] = sum(
                    meanAccNN) / len(meanAccNN)
                dictNN[f'precision promedio {count}:'] = sum(
                    meanPrecNN) / len(meanPrecNN)
                dictNN[f'recall promedio {count}:'] = sum(
                    meanRecallNN) / len(meanRecallNN)
                meanAUC = sum(
                    meanRocAucNN) / len(meanRocAucNN)
                if meanAUC > higherAuc:
                    higherAuc = meanAUC
                dictNN[f'roc Auc promedio {count}:'] = meanAUC
                dictNN[f'Loss promedio {count}:'] = sum(
                    meanLossNN) / len(meanLossNN)
                # metrics to desv. est.
                dictNN[f'accuracy Desv. Est {count}:'] = np.std(meanAccNN)
                dictNN[f'precision Desv. Est {count}:'] = np.std(meanPrecNN)
                dictNN[f'recall Desv. Est: {count}'] = np.std(meanRecallNN)
                dictNN[f'roc Auc Desv. Est: {count}'] = np.std(meanRocAucNN)
                dictNN[f'loss Desv. Est: {count}'] = np.std(meanLossNN)

                # mean of loss y epochs
                # print('epoch is ', epoch)
                # print('len bigger loss, ', len(meanLossNN))
                # print('loss bigger is ', meanLossNN)
                # print('sum bigger loss is,', sum(meanLossNN))
                if epoch == epochs[0]:
                    # print('im in 1')
                    meanLoss1.append(sum(meanLossNN) / len(meanLossNN))
                elif epoch == epochs[1]:
                    # print('im in 2')
                    meanLoss2.append(sum(meanLossNN) / len(meanLossNN))
                elif epoch == epochs[2]:
                    # print('im in 3')
                    meanLoss3.append(sum(meanLossNN) / len(meanLossNN))
                elif epoch == epochs[3]:
                    # print('im in 4')
                    meanLoss4.append(sum(meanLossNN) / len(meanLossNN))
                elif epoch == epochs[4]:
                    # print('im in 5')
                    meanLoss5.append(sum(meanLossNN) / len(meanLossNN))
                else:
                    # print('im in 6')
                    meanLoss6.append(sum(meanLossNN) / len(meanLossNN))
        # higher auc
        dictNN[f'\nHigher AUC of the model:'] = str(higherAuc)
        # calculate total mean
        # print('len loss is ', len(meanLoss1))
        # print('mean loss 1', meanLoss1)
        # print('mean loss 2', meanLoss2)
        # print('mean loss 3', meanLoss3)
        # print('mean loss 4', meanLoss4)
        # print('mean loss 5', meanLoss5)
        # print('mean loss 6', meanLoss6)
        meanLossList.append(sum(meanLoss1) / len(meanLoss1))
        meanLossList.append(sum(meanLoss2) / len(meanLoss2))
        meanLossList.append(sum(meanLoss3) / len(meanLoss3))
        meanLossList.append(sum(meanLoss4) / len(meanLoss4))
        meanLossList.append(sum(meanLoss5) / len(meanLoss5))
        meanLossList.append(sum(meanLoss6) / len(meanLoss6))
        # Return
        return dictNN, meanLossList, epochs

    # K-fold cross validation
    def bestConfig(self, completeDf, k, hidLayers, activFunc, epoch, lRate):
        # shuffle
        rowNum, _ = completeDf.shape
        completeDf = completeDf.sample(rowNum, random_state=1)
        # Target column and other columns of DataSet
        targ = completeDf.iloc[:, 0].to_numpy().astype(int)
        features = completeDf.iloc[:, 1:].to_numpy()

        # MLPClassifier
        # Neural Network Metrics
        # configuracion
        # topologia
        hiddenLayers = hidLayers
        activFunc = activFunc
        # Hiper parametros
        lr = lRate
        epochNum = epoch
        # Confusion matrix
        tnNN = 0
        fpNN = 0
        fnNN = 0
        tpNN = 0
        # VALUES FOR GRAPHICS
        tprs = []
        fprs = []
        precs = []
        recs = []
        # generate kFold splits
        kFold = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
        splitsIter = kFold.split(features, targ)
        # run neural network through all folds
        for trainIndex, testIndex in splitsIter:
            # train / test per fold
            X_train, X_test = features[trainIndex], features[testIndex]
            y_train, y_test = targ[trainIndex], targ[testIndex]

            # objeto clasificador
            clf = nn.MLPClassifier(
                # verbose=True,
                # batch_size=200,
                hidden_layer_sizes=hiddenLayers,
                # activation func default is relu
                activation=activFunc,
                early_stopping=False,
                solver='sgd',
                # learning_rate='constant',  # no change as epochs advance
                learning_rate_init=lr,
                max_iter=epochNum,
                random_state=1,  # To change random seed
                # tol=0.001  # minimum improvement in cost (tol)
                # number of epochs it was less than tol, to halt the process
                # n_iter_no_change=1000
            )
            # fit fold dataset
            clf.fit(X_train, y_train)
            # predict binary values
            y_pred = clf.predict(X_test)
            # predict probabilistic values
            y_prob = clf.predict_proba(X_test)
            # confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                y_test, y_pred).ravel()
            # print('conf matrix')
            # print('tn ', tn, 'fp ', fp,
            #       'fn ', fn, 'tp ', tp)
            tnNN += tn
            fpNN += fp
            fnNN += fn
            tpNN += tp
            # VALUES FOR GRAPHICS
            fpr, tpr, _ = roc_curve(
                y_test, y_prob[:, 1], drop_intermediate=False)
            precision, recall, _ = precision_recall_curve(
                y_test, y_prob[:, 1])
            # print('fpr is', fpr)
            # print('tpr is', tpr)
            # print('prec is', precision)
            # print('rec is', recall)
            fprs.append(fpr)
            tprs.append(tpr)
            precs.append(precision)
            recs.append(recall)
        # dictionary result of confussion matrix
        dictConfMat = {}
        dictConfMat['tn'] = tnNN
        dictConfMat['fp'] = fpNN
        dictConfMat['fn'] = fnNN
        dictConfMat['tp'] = tpNN
        # return
        return dictConfMat, fprs, tprs, precs, recs

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
