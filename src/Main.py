# Program to run test three different decission trees classification

# imports
from Coordinator import Coordinator
import pandas as pd

# Input
initPath = r'data\input\initial\DATA.csv'
# Output
initNormed = r'data\output\DATA_norm.txt'
resultsPath = r'data\output\Results.txt'


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
initPf = pd.read_csv(initPath, delimiter=';', encoding="ISO-8859-1")
normPf = coord.normalize(initPf)
print('normPf ', str(normPf))
writeToFile(str(normPf), initNormed, 'w')

# Run Spectral clustering for k 2 to k 10
coord.runConfig(normPf.sample(100))

# Run best k spectral clustering
resultDict = coord.run(normPf.sample(100), 6)
strResDict = dictToStr(resultDict)
writeToFile(strResDict, resultsPath, 'w')
