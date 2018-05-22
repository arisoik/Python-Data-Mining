import time
from functions import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

def createTripsCsv(trainsetFilepath, outputFilepath):
    isHeaderWrited = False
    tripId = 0

    if (os.path.exists(outputFilepath)):
        os.remove(outputFilepath)

    df = readCsv(trainsetFilepath, ',')
    uniqueVehicleIds = df.vehicleID.unique()
    for idx, vehicleId in enumerate(uniqueVehicleIds):
        vehicleJourneys = df.loc[df['vehicleID'] == vehicleId]

        vehicleJourneyPoints = []
        vehicleJourneyDf = pd.DataFrame()
        tempRef = vehicleJourneys.iloc[0]
        for row in vehicleJourneys.iterrows():
            if (row[1].journeyPatternId == tempRef.journeyPatternId):
                journeyPoint = [row[1].timestamp, row[1].longitude, row[1].latitude]
                vehicleJourneyPoints.append(journeyPoint)
            else:
                tripId += 1
                vehicleJourneyDf = vehicleJourneyDf.append(pd.DataFrame([{'TripId': tripId, 'JourneyPatternId': tempRef.journeyPatternId, 'JourneyPoints': vehicleJourneyPoints}], index=[0]), ignore_index=True)

                tempRef = row[1]
                vehicleJourneyPoints = []
                journeyPoint = [row[1].timestamp, row[1].longitude, row[1].latitude]
                vehicleJourneyPoints.append(journeyPoint)

        tripId += 1
        vehicleJourneyDf = vehicleJourneyDf.append(pd.DataFrame([{'TripId': tripId, 'JourneyPatternId': tempRef.journeyPatternId, 'JourneyPoints': vehicleJourneyPoints}], index=[0]), ignore_index=True)
        vehicleJourneyDf.to_csv(path_or_buf=outputFilepath, mode='a', header=not isHeaderWrited, index=False, columns=['TripId', 'JourneyPatternId', 'JourneyPoints'], sep=';')
        isHeaderWrited = True
        loading(idx + 1, len(uniqueVehicleIds))

def createTripsCleanCsv(tripsCsvFilepath, outputFilepath):
    tripsDf = readCsv(tripsCsvFilepath, ';')

    totalJourneys = len(tripsDf.index)
    tripsDf = tripsDf.drop(tripsDf[(tripsDf.JourneyPatternId == 'null') | (tripsDf.JourneyPatternId == '')].index) # https://stackoverflow.com/questions/13851535/how-to-delete-rows-from-a-pandas-dataframe-based-on-a-conditional-expression

    # Counters
    nullEmptyFilterCount = totalJourneys - len(tripsDf.index)
    lessPointsFilterCount = 0
    totalDistanceFilterCount = 0
    maxDistanceFilterCount = 0
    counter = 0
    for row in tripsDf.itertuples(): # https://stackoverflow.com/questions/7837722/what-is-the-most-efficient-way-to-loop-through-dataframes-with-pandas
        counter += 1
        journeyPoints = ast.literal_eval(row[3])
        if (len(journeyPoints) > 1):
            totalDistance = 0
            maxDistance = 0
            for i in range(len(journeyPoints)):
                if (i != len(journeyPoints) - 1):
                    currentDistance = haversine([journeyPoints[i][2], journeyPoints[i][1]], [journeyPoints[i + 1][2], journeyPoints[i + 1][1]], 0)
                    totalDistance += currentDistance
                    if (maxDistance < currentDistance):
                        maxDistance = currentDistance

            if (totalDistance < 2):
                totalDistanceFilterCount += 1
                tripsDf = tripsDf.drop(tripsDf[tripsDf.TripId == row[1]].index)
            elif (maxDistance > 2):
                maxDistanceFilterCount += 1
                tripsDf = tripsDf.drop(tripsDf[tripsDf.TripId == row[1]].index)
        else:
            lessPointsFilterCount += 1
            tripsDf = tripsDf.drop(tripsDf[tripsDf.TripId == row[1]].index)

        loading(counter, totalJourneys - nullEmptyFilterCount)

    tripsDf.to_csv(path_or_buf=outputFilepath, mode='w', header=True, index=False, columns=['TripId', 'JourneyPatternId', 'JourneyPoints'], sep=';')
    print
    print 'TripsCsvJourneys:' + str(totalJourneys) + ' | NullEmptyFilterCount: ' + str(nullEmptyFilterCount) + ' | ' + 'TotalDistanceFilterCount:' + str(totalDistanceFilterCount) + ' | MaxDistanceFilterCount: ' + str(maxDistanceFilterCount) + ' | LessPointsFilterCount: ' + str(lessPointsFilterCount) + ' | TripsCleanCsvJourneys: ' + str(len(tripsDf))

def findKNN(testCsvFilepath, tripsCleanCsvFilepath):
    tripsDf = readCsv(tripsCleanCsvFilepath, ';')
    testTripsDf = readCsv(testCsvFilepath, '\n')
    for testRow in testTripsDf.itertuples():
        outputDir = '../results/4.KNN_test_set_a1_(using_DTW)/Test Trip ' + str(testRow[0] + 1) + '/'
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        tripsColors = [[strPointsToLatLonArrays(testRow[1], 1), '#32CD32']]
        plotTrips(tripsColors, outputDir + 'Test Trip ' + str(testRow[0] + 1) + '.html')

        distancesDf = pd.DataFrame()
        for row in tripsDf.itertuples():
            tripJourneyPoints = ast.literal_eval(row[3])
            distance = DTWDistance(ast.literal_eval(testRow[1]), tripJourneyPoints)
            distancesDf = distancesDf.append(pd.DataFrame([{'DTW_Distance': distance, 'TripId': row[1]}], index=[0]), ignore_index=True)

        counter = 0
        distancesDf = distancesDf.sort_values(by='DTW_Distance').head(n=5)
        for row in distancesDf.itertuples():
            counter += 1
            neighborTripDf = tripsDf.loc[tripsDf['TripId'] == row[2]]
            tripsColors = [[strPointsToLatLonArrays(neighborTripDf['JourneyPoints'].iloc[0], 1), '#32CD32']]
            plotTrips(tripsColors, outputDir + 'N' + str(counter) + '_' + neighborTripDf['JourneyPatternId'].iloc[0] + '_' + '{0:.2f}'.format(row[1]) + 'km.html')

def findLCS(testCsvFilepath, tripsCleanCsvFilepath):
    tripsDf = readCsv(tripsCleanCsvFilepath, ';')
    testTripsDf = readCsv(testCsvFilepath, '\n')
    for testRow in testTripsDf.itertuples():
        outputDir = '../results/5.LCS_test_set_a2/Test Trip ' + str(testRow[0] + 1) + '/'
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        tripsColors = [[strPointsToLatLonArrays(testRow[1], 1), '#32CD32']]
        plotTrips(tripsColors, outputDir + 'Test Trip ' + str(testRow[0] + 1) + '.html')

        distancesDf = pd.DataFrame()
        for row in tripsDf.itertuples():
            tripJourneyPoints = ast.literal_eval(row[3])
            distance = LCSDistance(ast.literal_eval(testRow[1]), tripJourneyPoints)
            distancesDf = distancesDf.append(pd.DataFrame([{'LCS_Distance': distance, 'TripId': row[1]}], index=[0]), ignore_index=True)

        counter = 0
        distancesDf = distancesDf.sort_values(by='LCS_Distance', ascending=False).head(n=5)
        for row in distancesDf.itertuples():
            counter += 1
            neighborTripDf = tripsDf.loc[tripsDf['TripId'] == row[2]]
            tripsColors = []
            tripsColors.append([strPointsToLatLonArrays(neighborTripDf['JourneyPoints'].iloc[0], 1), '#32CD32'])
            tripsColors.append([strPointsToLatLonArrays(testRow[1], 1), '#D40404'])
            plotTrips(tripsColors, outputDir + 'N' + str(counter) + '_' + neighborTripDf['JourneyPatternId'].iloc[0] + '_Points#' + str(int(row[1])) + '.html')

def computeGridSequence(tripsFilepath, outputFilepath, ID_Title):
    trajectoryIndex = 0
    if(ID_Title == 'JourneyPatternId'):
        trajectoryIndex = 3
    else:
        trajectoryIndex = 2
    
    tripsDf = readCsv(tripsFilepath, ';')
    gridTripsDf = pd.DataFrame()
    for row in tripsDf.itertuples():
        tripJourneyPoints = ast.literal_eval(row[trajectoryIndex])
        gridStr = ''
        currGridStr = gridStr
        for i in range(len(tripJourneyPoints)):
            if (tripJourneyPoints[i][1] < minLon or tripJourneyPoints[i][1] > maxLon or tripJourneyPoints[i][2] < minLat or tripJourneyPoints[i][2] > maxLat):
                continue
            horizontalIndex = constrain(int(haversine([tripJourneyPoints[i][2], minLon], [tripJourneyPoints[i][2], tripJourneyPoints[i][1]], 0) / horizontalStep), 0, horizontalGrids-1)
            verticalIndex = constrain(int(haversine([minLat, tripJourneyPoints[i][1]], [tripJourneyPoints[i][2], tripJourneyPoints[i][1]], 0) / verticalStep), 0, verticalGrids-1)
            tempGridStr = 'C' + str(verticalIndex) + ',' + str(horizontalIndex) + ';'
            if (currGridStr != tempGridStr):
                currGridStr = tempGridStr
                gridStr = gridStr + currGridStr
        gridStr = gridStr[0:-1] # removing last ';'
        gridTripsDf = gridTripsDf.append(pd.DataFrame([{ID_Title: row[trajectoryIndex-1], 'GridSequence': gridStr}], index=[0]), ignore_index=True)

    gridTripsDf.to_csv(path_or_buf=outputFilepath, mode='w', header=True, index=False, columns=[ID_Title, 'GridSequence'], sep='\t')

def splitFeaturesAndLabels(dataCsv, labelCol):
    trainDf = readCsv(dataCsv, '\t')
    X = []
    y = np.array(trainDf[labelCol])
    for row in trainDf.itertuples():
        X.append(sumAxis(row[2], horizontalGrids, verticalGrids))
    X = np.array(X)
    return X, y

def findBestClassifier(X, y):
    knn = KNeighborsClassifier(n_neighbors=5)
    knnScores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    print 'k-Nearest Neighbors accuracy is: ' + str(knnScores.mean())
    maxAccuracy = knnScores.mean()
    classifier = knn
    classifierStr = 'k-Nearest Neighbors'    

    lr = LogisticRegression() 
    lrScores = cross_val_score(lr, X, y, cv=10, scoring='accuracy')
    print 'Logistic Regression accuracy is:' + str(lrScores.mean())
    if (lrScores.mean() > maxAccuracy):
        maxAccuracy = lrScores.mean()
        classifier = lr
        classifierStr = 'Logistic Regression'

    rf = RandomForestClassifier()
    rfScores = cross_val_score(rf, X, y, cv=10, scoring='accuracy')
    print 'Random Forest accuracy is: ' + str(rfScores.mean())
    if (rfScores.mean() > maxAccuracy):
        maxAccuracy = rfScores.mean()
        classifier = rf
        classifierStr = 'Random Forest'

    print 'Best classifier is: ' + classifierStr + 'with accuracy: ' + str(maxAccuracy)
    classifier = rf # comment this line to return the actual best classifier
    return classifier

    

# MAIN #
# Project's configurations
createProjectFolderStructure(['../results', '../results/1.First_Group_of_Data', '../results/2.Clean_Routes', '../results/3.Visualization', '../results/4.KNN_test_set_a1_(using_DTW)', '../results/5.LCS_test_set_a2', '../results/6.Features_gridTripsClean', '../results/7.Classification'])
trainsetFilepath = '../data_sets/train_set.csv'
tripsFilepath = '../results/1.First_Group_of_Data/trips.csv'
tripsCleanFilepath = '../results/2.Clean_Routes/tripsClean.csv'
tripsTestSetDTW = '../data_sets/test_set_a1.csv'
tripsTestSetLCSS = '../data_sets/test_set_a2.csv'
tripsCleanGridFilepath = '../results/6.Features_gridTripsClean/gridTripsClean.csv'
tripsTestFilepath = '../data_sets/test_set.csv'
tripsTestGridFilepath = '../results/7.Classification/gridTripsTest.csv'
predictionsFilepath = '../results/7.Classification/predictedTrips.csv'

os.system('cls')
os.system('clear')


# # Create trips.csv from raw data
print 'Processing Vehicles\' time/space stamps'
createTripsCsv(trainsetFilepath, tripsFilepath)
print
print


# # Create tripsClean.csv by "cleaning" trips.csv from "garbage" data
print 'Cleaning Trips'
createTripsCleanCsv(tripsFilepath, tripsCleanFilepath)
print


# # Visualization of first 5 JourneyPatternId from tripsClean.csv
tripsDf = readCsv(tripsCleanFilepath, ';')
selectedTripsDf = tripsDf.drop_duplicates(subset='JourneyPatternId').head(n=5)
for tripDf in selectedTripsDf.itertuples():
    tripsColors = [[strPointsToLatLonArrays(tripDf.JourneyPoints, 1), '#32CD32']]
    plotTrips(tripsColors, '../results/3.Visualization/journeyPatternID_' + str(tripDf.JourneyPatternId) + '.html')


# # Find k-Nearest Neighbors for given trajectories
print 'Finding k-Nearest Neighbors for given trajectories ...'
millis = int(round(time.time() * 1000))
findKNN(tripsTestSetDTW, tripsCleanFilepath)
print 'KNN_test_set_a1_(using_DTW) took ' + str((int(round(time.time() * 1000)) - millis)) + ' ms to finish'


# # Find 5 first-matching routes for given sub-routes with LCS method
print 'Finding 5 first-matching routes for given sub-routes with LCS method ...'
millis = int(round(time.time() * 1000))
findLCS(tripsTestSetLCSS, tripsCleanFilepath)
print 'LCS_test_set_a2 took ' + str((int(round(time.time() * 1000)) - millis)) +  'ms to finish'


# # Extract grid sequence for train/test datasets and then extract training features
print 'Extracting Features ...'
horizontalGrids, verticalGrids = (40, 40)
minLat, maxLat, minLon, maxLon = findMinMaxLatLon(tripsCleanFilepath)
horizontalStep = haversine([minLat, minLon], [minLat, maxLon], 0) / horizontalGrids
verticalStep = haversine([minLat, minLon], [maxLat, minLon], 0) / verticalGrids
computeGridSequence(tripsCleanFilepath, tripsCleanGridFilepath, 'JourneyPatternId') # comment it if horizontalGrids and verticalGrids are always the same
computeGridSequence(tripsTestFilepath, tripsTestGridFilepath, 'Test_Trip_ID') # comment it if horizontalGrids and verticalGrids are always the same
X, y = splitFeaturesAndLabels(tripsCleanGridFilepath, 'JourneyPatternId')
X_predict, _ = splitFeaturesAndLabels(tripsTestGridFilepath, 'GridSequence')


# # Classification. Find best classifier, fit with extracted features and predict JourneyPatternIDs for given trajectories
print 'Looking for the Best Classifier (best in accuracy) amongst: \n\tk-Nearest Neighbors\n\tLogistic Regression\n\tRandom Forest\n ... this may take a while ...\n'
millis = int(round(time.time() * 1000))
classifier = findBestClassifier(X, y)
classifier.fit(X, y)
pred = classifier.predict(X_predict)
predictionsDf = pd.DataFrame({'Test_Trip_ID': readCsv(tripsTestFilepath, ';')['Test_Trip_ID'], 'Predicted_JourneyPatternID': pred})
predictionsDf.to_csv(path_or_buf=predictionsFilepath, mode='w', header=True, index=False, columns=['Test_Trip_ID', 'Predicted_JourneyPatternID'], sep='\t')
print 'Classification took ' + str((int(round(time.time() * 1000)) - millis)) + ' ms to finish'
