import pandas as pd
import numpy as np

# Model Management
from comet_ml import Experiment

import sklearn

from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier


# Function to read in the dataset
def readData():
    # Read in the data from GitHub Repo
    data = pd.read_csv('https://raw.githubusercontent.com/aagupta/MentalAid/main/stats_train.csv')

    # Drop Unnamed column
    data2 = data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    return data2

# Function to split the dataset
def train(df):
    # Break out the dataframe into features and target variable
    df_len = df.shape[1] - 1
    X = df.iloc[:, 0:df_len]
    y = df.iloc[:, df_len]

    # Define the model
    etc = ExtraTreesClassifier(n_estimators=100, criterion='entropy')

    # Fit the model on the whole dataset
    etc.fit(X, y)

    # Computing the importance of each feature
    feature_importance = etc.feature_importances_

    # Normalizing the individual importances
    feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        etc.estimators_],
                                        axis = 0)
    
    fics = pd.DataFrame()
    fics['Feature'] = X.columns.tolist()
    fics['Importance'] = feature_importance_normalized
    
    fics.to_csv('/Users/aakritigupta/Desktop/Hackathon 2021/MentSea/MentSea/ETC_FeatureImportances.csv')
    

    # Evaluate the model
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    # n_scores = cross_val_score(etc, X, y, scoring='accuracy', cv=cv, n_jobs=1, error_score='raise')

    # Report the performance
    # acc = mean(n_scores)
    # print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    # Create an experiment with your api key
    # experiment = Experiment(
    #     api_key="qoJkmfBTZWvvgzCerRG4WTivF",
    #     project_name="MentalAid_ETC",
    #     workspace="aakritigupta",
    # )

    # # Log the following values
    # experiment.log_metric("accuracy", acc)

    return etc

# Function to predict on the new validation dataset
def val(df, modelObj):
    # Read in the new dataset
    valDf = pd.read_csv('/Users/aakritigupta/Desktop/Hackathon 2021/MentSea/MentSea/Validation_2019.csv')

    # Drop unnecessary columns
    valDf2 = valDf.drop(columns=['Unnamed: 0'])

    # Predict on the new dataset
    new_pred = modelObj.predict(valDf2)
    # print(new_pred)

    # Add the predictions to the new dataset
    valDf2['Predictions'] = new_pred

    # Write the df to a CSV
    valDf2.to_csv('/Users/aakritigupta/Desktop/Hackathon 2021/MentSea/MentSea/Predictions_2019.csv')

    # return valDf

# Function to run full script
def main():
    dataDf = readData()
    # print(dataDf.shape)
    # print(dataDf.head())
    etcObj = train(dataDf)
    # test = val(dataDf, etcObj)
    # print(test.columns)


main()