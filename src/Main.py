# Program to run test three different decission trees classification

# imports
from Coordinator import Coordinator
import pandas as pd

# Input
initPath = r'data\input\initial\DATA.csv'
# Output
initNormed = r'data\output\DATA_norm.txt'
bestDictOut = r'data\output\BestDictionary.txt'
tablaResOutput = r'data\output\TableResults.txt'
cartOutput = r'data\output\Cart.txt'
c45Output = r'data\output\C45.txt'


# function to write trazability of program to txt file
def writeToFile(data, path, action):
    data = str(data)
    with open(path, action, encoding="utf-8") as f:  # open the file to write
        f.write(data)


# function to pass dictionary to text
def dictToStr(dict):
    strDict = ''
    for k, v in dict.items():
        strDict += f'{str(k)} -> {str(v)}\n'
    return strDict


# Join dataSets and normalize
coord = Coordinator()
# joinedDS = coord.join(pathA, pathB)
# normalize joined Dataset
initPf = pd.read_csv(initPath, delimiter=',')
normPf = coord.normalize(initPf)
writeToFile(str(normPf), initNormed, 'w')

# # Call 5 NN configurations
# k = 10
# # 1st config
# hid1 = (16, 8, 6)  # (120, 70, 40, 20)
# activF1 = 'relu'
# resDict1, meanLossList1, epochs1 = coord.runConfig(normPf, k, hid1, activF1)
# strDict1 = '1st Configuration NN:\n' + dictToStr(resDict1)
# writeToFile(strDict1, tablaResOutput, 'w')
# # 2nd config
# hid2 = (30, 18, 10, 4)  # (60, 36, 26, 20, 15)
# activF2 = 'relu'
# resDict2, meanLossList2, epochs2 = coord.runConfig(normPf, k, hid2, activF2)
# strDict2 = '\n\n\n2nd Configuration NN:\n' + dictToStr(resDict2)
# writeToFile(strDict2, tablaResOutput, 'a')
# # 3rd config
# hid3 = (24, 16, 8, 2)  # (88, 62, 33, 12)
# activF3 = 'tanh'
# resDict3, meanLossList3, epochs3 = coord.runConfig(normPf, k, hid3, activF3)
# strDict3 = '\n\n\n3rd Configuration NN:\n' + dictToStr(resDict3)
# writeToFile(strDict3, tablaResOutput, 'a')
# # 4th config
# hid4 = (36, 20, 6)  # (99, 50, 30)
# activF4 = 'relu'
# resDict4, meanLossList4, epochs4 = coord.runConfig(normPf, k, hid4, activF4)
# strDict4 = '\n\n\n4th Configuration NN:\n' + dictToStr(resDict4)
# writeToFile(strDict4, tablaResOutput, 'a')
# # 5th config
# hid5 = (44, 20)  # (108, 20)
# activF5 = 'tanh'
# resDict5, meanLossList5, epochs5 = coord.runConfig(normPf, k, hid5, activF5)
# strDict5 = '\n\n\n5th Configuration NN:\n' + dictToStr(resDict5)
# writeToFile(strDict5, tablaResOutput, 'a')

# # Call best configuration
# hidBest = (24, 16, 8, 2)  # (120, 70, 40, 20)
# activFBest = 'relu'
# epochBest = 700
# lrBest = 0.1
# bestConfigDict, fprs, tprs, precs, recs = coord.bestConfig(
#     normPf, k, hidBest, activFBest, epochBest, lrBest)
# strBestDict = 'Best NN Configuration Confussion Matrix:\n' + \
#     dictToStr(bestConfigDict)
# writeToFile(strBestDict, bestDictOut, 'w')

# # Graphs for best configuration
# coord.callGraphics(fprs, tprs, precs, recs, epochs1, meanLossList1)
