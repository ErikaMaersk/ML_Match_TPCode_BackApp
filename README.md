# ML_Match_TPCode_BackApp
This repository contains code and documentation related to an initiative to predict backend applications associated with OpenText TPCodes. 

Objective - Match OpenText TP Codes to Backend Applications so that we can better identify and quantify OpenText's place in the wider Maersk technical landscape.

There is a nomenclature associated with OpenText TPCodes that provides insight into business area, direction/processing, trading partner and message type. The volume data Opentext provide monthly also includes Partner name, Partner Name for Reports and Flow ID (Axway identifier). Axway documentation provides a link between the TP Code to Flow ID in the volume reports and the Flow ID to backend application in the Axway documentation. The features of the TP Codes that have been matched to backend applications are used to to train a Random Forest ML model to predict the backend applications of the TP Codes without matches (and without Flow IDs).
Inputs:
Raw_Train_Test_Validate_Dataset - A csv file containing the complete data (i.e. TP Code, additional charactaristics/features and the associated backend application)
OpenText Backend Application Prediction - A csv file with only TP Code and additional charactaristics/features but no associated backend application

Step One: Run the 'OpenText_Backend_Data_Split' programme to 'shuffle' the observations in the Raw_Train_Test_Validate_Dataset and divide them into training, testing and validation datasets. Outputs: three files -> 'test.csv', 'train.csv' and 'validation.csv'.

Step Two: Run the 'OpenText_Backend_Model_Train' programme to train the model on the 'train.csv' file, validate the model on the 'validation.csv' file and produce a 'report card' of the model's performance. Output: 'validation_predictions_final.csv' which serves as a report card of the models performance on the 'validation.csv' file. 

Step Three: Run the 'OpenText_Backend_Model_Predict' programme to apply the model to the 'OpenText Backend Application Prediction.csv' file where the Backend Applications are unknown. Output: 'backend_predictions_output.csv' which lists the TP Codes matched to three backend applications with the correspond confidence value and flags. The data in this file can then be used in a VLOOKUP.
