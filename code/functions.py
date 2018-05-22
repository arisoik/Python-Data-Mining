import math, gmplot, os, ast
from math import radians, cos, sin, asin, sqrt, fabs
import numpy as np
import pandas as pd
import sys, time

def createProjectFolderStructure(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def readCsv(filepath, sep):
    df = pd.read_csv(filepath_or_buffer=filepath, keep_default_na=False, sep=sep)  # https://stackoverflow.com/questions/44128033/pandas-reading-null-as-a-nan-float-instead-of-str
    return df

def haversine(latLon1, latLon2, latIndex): # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lat1 = latLon1[latIndex]
    lon1 = latLon1[latIndex+1]
    lat2 = latLon2[latIndex]
    lon2 = latLon2[latIndex+1]

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371.0 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def DTWDistance(x, y): # https://en.wikipedia.org/wiki/Dynamic_time_warping
    m = len(x)
    n = len(y)
    DTW = np.zeros((m+1, n+1))

    # fill the first row of the array
    for i in range(n+1):
        DTW[0, i] = float('inf')
    # fill the first column of the array
    for i in range(m+1):
        DTW[i, 0] = float('inf')
    DTW[0, 0] = 0    

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = haversine(x[i-1], y[j-1], 1)
            DTW[i, j] = cost + min([DTW[i-1, j], DTW[i, j-1], DTW[i-1, j-1]])

    return DTW[m, n]

def LCSDistance(x, y): # https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
    m = len(x)
    n = len(y)
    C = np.zeros((m+1, n+1))

    # fill the first row of the array
    for i in range(n+1):
        C[0, i] = 0
    # fill the first column of the array
    for i in range(m+1):
        C[i, 0] = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            if(haversine(x[i-1], y[j-1], 1) <= 0.2): 
                C[i, j] = C[i-1, j-1] + 1
            else:
                C[i, j] = max([C[i, j-1], C[i-1, j]])
    
    return C[m, n]

def findMinMaxLatLon(tripsCleanCsvFilepath):
    tripsDf = readCsv(tripsCleanCsvFilepath, ';')
    tripJourneyPoints = ast.literal_eval(tripsDf['JourneyPoints'].iloc[0])
    minLat = tripJourneyPoints[0][2]
    maxLat = tripJourneyPoints[0][2]
    minLon = tripJourneyPoints[0][1]
    maxLon = tripJourneyPoints[0][1]
    for row in tripsDf.itertuples():
        tripJourneyPoints = ast.literal_eval(row[3])
        for i in range(len(tripJourneyPoints)):
            minLat = min([minLat, tripJourneyPoints[i][2]])
            maxLat = max([maxLat, tripJourneyPoints[i][2]])
            minLon = min([minLon, tripJourneyPoints[i][1]])
            maxLon = max([maxLon, tripJourneyPoints[i][1]]) 
    return minLat, maxLat, minLon, maxLon

def constrain(val, min_val, max_val): # https://stackoverflow.com/questions/34837677/a-pythonic-way-to-write-a-constrain-function/34837691
    return min(max_val, max(min_val, val))

def strPointsToLatLonArrays(strPoints, lonIndex):
    latitudes = []
    longitudes = []
    journeyPoints = ast.literal_eval(strPoints)
    for i in range(len(journeyPoints)):
        latitudes.append(journeyPoints[i][lonIndex+1])
        longitudes.append(journeyPoints[i][lonIndex])
    return latitudes, longitudes

def plotTrips(tripsColors, outputFilepath):
    for i, tripColor in enumerate(tripsColors):
        latitudes = tripColor[0][0]
        longitudes = tripColor[0][1]
        color = tripColor[1]
        if(i==0):
            gmap = gmplot.GoogleMapPlotter(latitudes[0], longitudes[0], 14)
        gmap.plot(latitudes, longitudes, color, edge_width=5)
    gmap.draw(outputFilepath)


def startEndLength(gridStr):
    gridPoints = gridStr.split(';')

    start = gridPoints[0][1:]
    end = gridPoints[-1][1:]
    Yi, Xi = start.split(',')
    Yo, Xo = end.split(',')

    Xs = []
    Ys = []
    for gridPoint in gridPoints:
        point = gridPoint[1:]
        point = point.split(',')
        Ys.append(int(point[0]))
        Xs.append(int(point[1]))
    length = 0.0
    for i in range(1, len(Ys)):
        dy = fabs(Ys[i] - Ys[i-1])
        dx = fabs(Xs[i] - Xs[i-1])
        if dx == 0.0:
            length = length + dy
        elif dy == 0.0:
            length = length + dx
        else:
             length = length + sqrt(dx*dx + dy*dy)
    return [int(Xi), int(Yi), int(Xo), int(Yo), length]


def sumAxis(gridStr, horizontalGrids, verticalGrids):
    featuresArray = np.zeros((horizontalGrids + verticalGrids + 5), int) # vertical grids are first
    gridPoints = gridStr.split(';')
    for gridPoint in gridPoints:
        point = gridPoint[1:]
        point = point.split(',')
        featuresArray[int(point[0])] += 1
        featuresArray[verticalGrids + int(point[1])] += 1
    
    featuresArray[-5], featuresArray[-4], featuresArray[-3], featuresArray[-2], featuresArray[-1] = startEndLength(gridStr) # comment this line to have only sumAxis functionality, otherwise you combine it with startEndLength
    return featuresArray


def slopesStartEndWidthHeight(gridStr):
    featuresArray = np.zeros(14, int) # features [0, 45, 90, 135, 180, 225, 270, 315, Xi, Yi, Xo, Yo, width, height]
    gridPoints = gridStr.split(';')
    for i in range(len(gridPoints)-1):
        currentPoint = gridPoints[i][1:]
        currentPoint = currentPoint.split(',')
        nextPoint = gridPoints[i+1][1:]
        nextPoint = nextPoint.split(',')
        slopeDeg = int(math.degrees(math.atan2((int(nextPoint[0]) - int(currentPoint[0])), (int(nextPoint[1]) - int(currentPoint[1]))))) % 360
        if(slopeDeg < 45):
            featuresArray[0] = featuresArray[0] + 1
        elif(slopeDeg < 90):
            featuresArray[1] = featuresArray[1] + 1
        elif(slopeDeg < 135):
            featuresArray[2] = featuresArray[2] + 1
        elif(slopeDeg < 180):
            featuresArray[3] = featuresArray[3] + 1
        elif(slopeDeg < 225):
            featuresArray[4] = featuresArray[4] + 1
        elif(slopeDeg < 270):
            featuresArray[5] = featuresArray[5] + 1
        elif(slopeDeg < 315):
            featuresArray[6] = featuresArray[6] + 1
        elif(slopeDeg < 360):
            featuresArray[7] = featuresArray[7] + 1
    
    start = gridPoints[0][1:]
    end = gridPoints[-1][1:]
    Yi, Xi = start.split(',')
    Yo, Xo = end.split(',')
    featuresArray[8] = int(Xi)
    featuresArray[9] = int(Yi)
    featuresArray[10] = int(Xo)
    featuresArray[11] = int(Yo)
    
    minY = int(Yi)
    maxY = minY
    minX = int(Xi)
    maxX = minX
    for gridPoint in gridPoints:
        point = gridPoint[1:]
        point = point.split(',')
        minY = min(minY, int(point[0]))
        maxY = max(maxY, int(point[0]))
        minX = min(minX, int(point[1]))
        maxX = max(maxX, int(point[1]))
    featuresArray[12] = maxX - minX
    featuresArray[13] = maxY - minY
    return featuresArray


def bagOfWords(gridStr, horizontalGrids, verticalGrids):
    featuresArray = np.zeros((horizontalGrids * verticalGrids), int)
    gridElements = gridStr.split(';')
    for element in gridElements:
        element = element[1:]
        temp = element.split(',')
        vIndex, hIndex = (int(temp[0]), int(temp[1]))
        featuresArray[vIndex*horizontalGrids + hIndex] += 1
    return featuresArray

def loading(index, total):
    percent = (index * 100) / total
    rollingSymbol = findRollingSymbol(index)
    x = '[' + rollingSymbol + '] ' + str(percent) + '%\t' + printProgressBar(percent)
    sys.stdout.write('\r' + x)
    sys.stdout.flush()

def findRollingSymbol(index):
    symbol = ''
    if (index % 4 == 1):
        symbol = '/'
    if (index % 4 == 2):
        symbol = '-'
    if (index % 4 == 3):
        symbol = '\\'
    if (index % 4 == 0):
        symbol = '|'
    return symbol
        
def printProgressBar(percent):
    bar = '[                                                                                                    ]'
    if (percent == 0):
        return bar
    for i in range(1, percent+1):
        bar = bar[:i] + unichr(0x2588) + bar[i+1:]
    return bar
