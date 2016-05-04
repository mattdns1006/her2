import openslide
import pandas as pd
#import matplotlib.pyplot as plt
import glob
import re
import pdb

groundTruth = pd.read_csv("data/groundTruth.csv")
filePaths = glob.glob("data/*.ndpi")

def getPair(caseNumber,level):
    # X 
    obs = openslide.open_slide('data/'+str(caseNumber)+'_HER2.ndpi')
    dims = obs.level_dimensions[level]
    print("Dimensions ", dims)
    x = obs.read_region((0,0),level,(dims[0],dims[1]))

    # Y 
    case = groundTruth.loc[groundTruth["CaseNo"]==caseNumber]
    y1, y2 = case["HeR2 SCORE"].values[0], case["PERCENTAGE CELLS WITH COMPLETE MEMBRANE STAINING IRRESPECTIVE OF INTENSITY"].values[0]
    return x, y1, y2

def saveJPG(filePath,level): #loads Whole slide image (WSI) and saves as JPG given level of resolution
    name = filePath.split("/")[1].split(".")[0]
    obs = openslide.open_slide(filePath)
    dims = obs.level_dimensions[level]
    x = obs.read_region((0,0),level,(dims[0],dims[1]))
    x.save("data/cleaned/"+name+".jpg")
    print("saved")

if __name__ == "__main__":
    #pdb.set_trace()
    for filePath in filePaths:
        saveJPG(filePath,6)
        print(filePath)
