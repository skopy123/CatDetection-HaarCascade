#this file is not standalone program, it is module used by CatAlarm.
#this module loads cat database from file, keeps tensor for all templates in memory and 
#provides functions to compare unknown image tensor to database
#it does not do any ML inference, it just loads tensors from file and compares them

import os
import json
import numpy as np

from scipy.spatial.distance import cosine
from cat_info import CatInfo
from catMatchStats import CatMatchStats
import modSettings as settings


print("Loading Cat Templates database")


dbFileName = settings.baseFolder + "catsDB" 
if settings.useColorNormalization:
    dbFileName += "_wCN"
if settings.useModelWithReg:
    dbFileName += "_wReg"
dbFileName += ".json"


if not os.path.isfile(dbFileName):
    print("Cat database file not found, please run TemplateDatabaseBuilder.py first")
    exit()

#load cats from file
catInfoArrayAll = []
with open(dbFileName) as json_file:
    jsonArr = json.load(json_file)
    for jsonObj in jsonArr:
        catInfoArrayAll.append(CatInfo.FromJson(jsonObj))
        #print short description and tensor size
        print (catInfoArrayAll[-1].ShortDescription(), " length ", len(catInfoArrayAll[-1].outputTensor))
    print("Loaded ", len(jsonArr), " cats from file")

#define compute cosine distance between two catInfo objects
def computeDistance(catInfo1, catInfo2):
    return cosine(catInfo1.outputTensor, catInfo2.outputTensor)

#order array by catName
catInfoArrayAll.sort(key=lambda x: x.catName, reverse=False)
#filter out night vision images
catInfoArrayDayMode = [x for x in catInfoArrayAll if x.nightVision == False]
#filter out day vision images
catInfoArrayNightMode = [x for x in catInfoArrayAll if x.nightVision == True]


catNames = []#["Amalka", "Gita", "Kocour", "Maugli"]
for catInfo in catInfoArrayAll:
    if not catInfo.catName in catNames:
        catNames.append(catInfo.catName)
totalCatCount = len(catNames)

def printDatabaseStats():
    print("Database stats:")
    print("Stats for day mode:")
    printDatabaseStatsSingleSet(catInfoArrayDayMode)
    print("Stats for night vision mode:")
    printDatabaseStatsSingleSet(catInfoArrayNightMode)


def printDatabaseStatsSingleSet(catInfos):
    print("  Number of cats template: ", len(catInfos))

    #create table with computeDistance between all cats in set
    distanceTable = np.zeros((len(catInfos), len(catInfos)))
    for i in range(len(catInfos)):
        for j in range(len(catInfos)):
            distanceTable[i][j] = computeDistance(catInfos[i], catInfos[j])

    #print distance table formatted to command line
    print("Distance table:")
    print("      ", end="")
    for catInfo in catInfos:
        print("{:9s}".format(catInfo.ShortDescription()), end="")
    print("")
    for i in range(len(catInfos)):
        print("{:9s}".format(catInfos[i].ShortDescription()), end="")
        for j in range(len(catInfos)):
            #print value in color from green to red depending on distance
            distance = distanceTable[i][j]
            #compute color based on distance. 0 is green, 1 is red
            vcc = int(127 * (distance / 1))
            if vcc > 127:
                vcc = 127
            if vcc < 0:
                vcc = 0
            color = (127+vcc, 127-vcc, 127)
            #print("{:12.2f}".format(), end="")
            print("\033[38;2;{};{};{}m{:9.2f}\033[0m".format(color[2], color[1], color[0], distance), end="")
        print("")
    print("")
    print("")

def compareUnknownTensorToDB(unknownTensor, nvMode):
    #prepare results collection
    results = []
    for i in range(0,totalCatCount):
        cms = CatMatchStats()
        cms.Name = catNames[i]
        results.append(cms)
    
    #prepare template array
    activeCatInfoArray = catInfoArrayDayMode
    if nvMode:
        activeCatInfoArray = catInfoArrayNightMode

    #compute distance to all templates
    for i in range(0,activeCatInfoArray.__len__()):
        tensorDist = cosine(unknownTensor, activeCatInfoArray[i].outputTensor)
        catIndex = catNames.index(activeCatInfoArray[i].catName)
        results[catIndex].DistanceToTemplates.append(tensorDist)
    
    #compute min and avg distance
    for i in range(0,totalCatCount):
        results[i].DistanceToTemplatesMin = min(results[i].DistanceToTemplates)
        results[i].DistanceToTemplatesAvg = sum(results[i].DistanceToTemplates) / len(results[i].DistanceToTemplates)

    #sort results by distance
    return results.sort(key=lambda x: x.DistanceToTemplatesAvg, reverse=True)