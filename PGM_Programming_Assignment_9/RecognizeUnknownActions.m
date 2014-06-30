% You should put all your code for recognizing unknown actions in this file.
% Describe the method you used in YourMethod.txt.
% Don't forget to call SavePrediction() at the end with your predicted labels to save them for submission, then submit using submit.m

load('PA9Data.mat');

actionNum = length(datasetTest3.actionData);
datasetTest3.labels = ones(actionNum, 1);

[accuracy, predictions] = RecognizeActions(datasetTrain3, datasetTest3, G, 2);

SavePredictions(predictions);




