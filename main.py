from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import randomForest, adaBoost
import pandas as pd


def trainingDataPreprocessingAndFeatureEngineering(file):
    ## Load data and format it
    csv_file_object = csv.reader(open(file)) # Load in the csv file
    header = next(csv_file_object) 					  # Skip the fist line as it is a header
    data=[] 											  # Create a variable to hold the data

    for row in csv_file_object: # Skip through each row in the csv file,
        data.append(row) 	# adding each row to the data variable

    for i in range(len(data)):
        data[i] = data[i][0].split(',')

    for l in data:
        l[0] = float(l[0])
        l[1] = float(l[1])
        l[2] = float(l[2])
        l.pop(3)
        l[3] = l[3].split()[0]
        if l[4] == 'male':
            l[4] = 1.
        else:
            l[4] = 2.
        if l[5] != '':
            l[5] = float(l[5])
        else:
            l[5] = ''
        l[6] = float(l[6])
        l[7] = float(l[7])
        l[9] = float(l[9])

    headers = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

    X = pd.DataFrame(data, columns = headers)


    ## data preprocessing

    #useless features : passengerId, Ticket
    del X['PassengerId']
    del X['Ticket']
    #feature Cabin
    X['Cabin'].value_counts() ## 686 of 890 are not given : useless
    del X['Cabin']
    #Let's have a look at how many values are missing for each feature
    X['Name'].value_counts() #0 value missing
    X['Sex'].value_counts() # 0 value missing
    X['Pclass'].value_counts() #0 value missing
    X['Age'].value_counts(sort = True) #177 values missing : let's give the average value to all the missing values
    X['SibSp'].value_counts() #0 value missing
    X['Parch'].value_counts() #0 value missing
    X['Fare'].value_counts(sort = True) #0 value missing
    X['Embarked'].value_counts() #2 values missing.. can't do anything about it.. let's delete the two data points

    X = X[X.Embarked != '']

    sum = 0
    c = 0
    for i in X['Age']:
        if i != '':
            c +=1
            sum += i
    mean_age = sum/c

    X['Age'][X['Age']==''] = float(mean_age)

    y = X['Survived'] # Save labels to y

    del X['Survived'] # Remove survival column from matrix X

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    X["Embarked"] = le.fit_transform(X["Embarked"].fillna('0'))
    X["Name"] = le.fit_transform(X["Name"].fillna('0'))

    # Add a vulnerability feature based on age
    X['Vulnerability'] = pd.Series(random.randn(X.shape[0]))
    X['Vulnerability'][X['Age'] < 15] = (15. - X['Age'])/15.
    X['Vulnerability'][X['Age'] > 50] = (X['Age'] - 50.)/50.
    X['Vulnerability'][(X['Age'] >= 15) & (X['Age'] <= 50)] = 0.

    print(X)

    #Let's normalize our features values

    X = (X - X.mean())/X.std()

    #Feature Selection : only keep 4  best features
    del X['Name']
    del X['Embarked']
    del X['SibSp']
    del X['Parch']
    del X['Age']
    #del X['Vulnerability']

    return X, y


def testDataPreprocessingAndFeatureEngineering(file):
    ## Load data and format it
    csv_file_object = csv.reader(open(file)) # Load in the csv file
    header = next(csv_file_object) 					  # Skip the fist line as it is a header
    data=[] 											  # Create a variable to hold the data

    for row in csv_file_object: # Skip through each row in the csv file,
        data.append(row) 	# adding each row to the data variable

    for i in range(len(data)):
        data[i] = data[i][0].split(',')


    for l in data:
        l[0] = float(l[0])
        l[1] = float(l[1])
        l.pop(2)
        l[2] = l[2].split()[0]
        if l[3] == 'male':
            l[3] = 1.
        else:
            l[3] = 2.
        if l[4] != '':
            l[4] = float(l[4])
        else:
            l[4] = ''
        l[5] = float(l[5])
        l[6] = float(l[6])
        if (l[8] != ''):
            l[8] = float(l[8])
        else:
            l[8] = 0.


    headers = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

    X = pd.DataFrame(data, columns = headers)


    ## data preprocessing

    #useless features : passengerId, Ticket
    del X['PassengerId']
    del X['Ticket']
    #feature Cabin
    X['Cabin'].value_counts() ## 686 of 890 are not given : useless
    del X['Cabin']
    #Let's have a look at how many values are missing for each feature
    X['Name'].value_counts() #0 value missing
    X['Sex'].value_counts() # 0 value missing
    X['Pclass'].value_counts() #0 value missing
    X['Age'].value_counts(sort = True) #177 values missing : let's give the average value to all the missing values
    X['SibSp'].value_counts() #0 value missing
    X['Parch'].value_counts() #0 value missing
    X['Fare'].value_counts(sort = True) #0 value missing
    X['Embarked'].value_counts() #2 values missing.. can't do anything about it.. let's delete the two data points

    X = X[X.Embarked != '']

    sum = 0
    c = 0
    for i in X['Age']:
        if i != '':
            c +=1
            sum += i
    mean_age = sum/c

    X['Age'][X['Age']==''] = float(mean_age)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    X["Embarked"] = le.fit_transform(X["Embarked"].fillna('0'))
    X["Name"] = le.fit_transform(X["Name"].fillna('0'))

    # Add a vulnerability feature based on age
    X['Vulnerability'] = pd.Series(random.randn(X.shape[0]))
    X['Vulnerability'][X['Age'] < 15] = (15. - X['Age'])/15.
    X['Vulnerability'][X['Age'] > 50] = (X['Age'] - 50.)/50.
    X['Vulnerability'][(X['Age'] >= 15) & (X['Age'] <= 50)] = 0.

    print(X)

    #Let's normalize our features values

    X = (X - X.mean())/X.std()

    #Feature Selection : only keep 4  best features
    del X['Name']
    del X['Embarked']
    del X['SibSp']
    del X['Parch']
    del X['Age']
    #del X['Vulnerability']

    return X

##Training

def training(X, y, D1, T1, D2, T2):

    # Initialize cross validation
    kf = cross_validation.KFold(X.shape[0], n_folds=10)

    ### Bagging Random Forest

    print('Random Forest classifier')

    totalInstances = 0 # Variable that will store the total intances that will be tested
    totalCorrect = 0 # Variable that will store the correctly predicted intances

    for trainIndex, testIndex in kf:
        trainSet = take(X, trainIndex, axis = 0)
        testSet = take(X, testIndex, axis = 0)
        trainLabels = take(y, trainIndex, axis = 0)
        testLabels = take(y, testIndex, axis = 0)

        predictedLabels = randomForest(trainSet, trainLabels, testSet, D1, T1)

        correct = 0
        for i in range(testSet.shape[0]):
            if predictedLabels[i] == testLabels[i]:
                correct += 1

        print('Accuracy: ' + str(float(correct)/(testLabels.shape[0])))
        totalCorrect += correct
        totalInstances += testLabels.size
    totalAccuracyBagging = totalCorrect/float(totalInstances)
    print('Total Accuracy: ' + str(totalAccuracyBagging))

    print('Adaboost classifier')

    totalInstances = 0 # Variable that will store the total intances that will be tested
    totalCorrect = 0 # Variable that will store the correctly predicted intances

    for trainIndex, testIndex in kf:
        trainSet = take(X, trainIndex, axis = 0)
        testSet = take(X, testIndex, axis = 0)
        trainLabels = take(y, trainIndex, axis = 0)
        testLabels = take(y, testIndex, axis = 0)

        predictedLabels = adaBoost(trainSet, trainLabels, testSet, D2, T2)

        correct = 0
        for i in range(testSet.shape[0]):
            if predictedLabels[i] == testLabels[i]:
                correct += 1

        print('Accuracy: ' + str(float(correct)/(testLabels.shape[0])))
        totalCorrect += correct
        totalInstances += testLabels.size
    totalAccuracyBoosting = totalCorrect/float(totalInstances)
    print('Total Accuracy: ' + str(totalAccuracyBoosting))

    return totalAccuracyBagging, totalAccuracyBoosting


X_train, y_train = trainingDataPreprocessingAndFeatureEngineering('Data/train.csv')
X_test = testDataPreprocessingAndFeatureEngineering('Data/test.csv')

###### Training with cross-validation
training(X_train.as_matrix(),y_train.as_matrix(),4, 2000, 1, 1057)

##### Compare the accuracies with different values of maximal depth and number of trees, with cross validation
##### Only execute this part if you want to plot the 3D graphs, but it takes a long time

#depth = array([1,2,3,4])
#nbTrees = array([25,50,75,100,250,500,1000])

#baggingAccuracies  = zeros((depth.shape[0],nbTrees.shape[0]))
#boostingAccuracies  = zeros((depth.shape[0],nbTrees.shape[0]))

#for a in range(baggingAccuracies.shape[0]):
#    for b in range(boostingAccuracies.shape[1]):
#        baggingAccuracies[a,b] = training(X_train.as_matrix(), y_train.as_matrix(), depth[a], nbTrees[b], depth[a], nbTrees[b])[0]
#        boostingAccuracies[a,b] = training(X_train.as_matrix(), y_train.as_matrix(), depth[a], nbTrees[b], depth[a], nbTrees[b])[1]

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


#fig = plt.figure()
#ax = plt.axes(projection='3d')
#d, t = meshgrid(depth, nbTrees)
#ax.plot_surface(d.T, t.T, baggingAccuracies, color='blue')
#ax.set_title('Total Accuracy Bagging vs Depth and number of trees')
#ax.set_xlabel('max_depth')
#ax.set_ylabel('number of trees')
#ax.set_zlabel('Total Accuracy with cross-validation');
#plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#d, t = meshgrid(depth, nbTrees)
#ax.plot_surface(d.T, t.T, boostingAccuracies, color='red')
#ax.set_title('Total Accuracy Bagging vs Depth and number of trees')
#ax.set_xlabel('max_depth')
#ax.set_ylabel('number of trees')
#ax.set_zlabel('Total Accuracy with cross-validation');
#plt.show()



##### Prediction work
def predictBagging(trainSet, trainLabels, testSet):

    print('Random Forest classifier')

    predictedLabelsBagging = randomForest(trainSet, trainLabels, testSet, 4, 2000)
    return predictedLabelsBagging


def predictBoosting(trainSet, trainLabels, testSet):

    print('Adaboost classifier')

    predictedLabelsBoosting = adaBoost(trainSet, trainLabels, testSet, 1, 1057)
    return predictedLabelsBoosting

baggingPrediction = predictBagging(X_train, y_train, X_test)
boostingPrediction = predictBoosting(X_train, y_train, X_test)

baggingPredictionData = {
    'PassengerId' : [i for i in range(892, 892 + len(baggingPrediction))],
    'Survived' : baggingPrediction.astype(int)
}
boostingPredictionData = {
    'PassengerId' : [i for i in range(892, 892 + len(baggingPrediction))],
    'Survived' : boostingPrediction.astype(int)
}

## Strategy here : there are statistically more zeros than ones : if the two models agree, we keep the predicted value, else we put 0

ensemblePrediction = zeros(baggingPrediction.shape)


differences = 0
for i in range(len(baggingPrediction)):
    if baggingPrediction[i] == boostingPrediction[i]:
        ensemblePrediction[i] = baggingPrediction[i]
    else:
        ensemblePrediction[i] = 0
        differences +=1

print(differences/ensemblePrediction.shape[0])

ensemblePredictionData = {
    'PassengerId' : [i for i in range(892, 892 + len(baggingPrediction))],
    'Survived' : ensemblePrediction.astype(int)
}


baggingPredictionDf = pd.DataFrame(data=baggingPredictionData)
boostingPredictionDf = pd.DataFrame(data=boostingPredictionData)
ensemblePredictionDf = pd.DataFrame(data=ensemblePredictionData)

baggingPredictionDf.to_csv('baggingPrediction.csv', sep=',', index= False)
boostingPredictionDf.to_csv('boostingPrediction.csv', sep=',', index= False)
ensemblePredictionDf.to_csv('ensemblePrediction.csv', sep=',', index= False)