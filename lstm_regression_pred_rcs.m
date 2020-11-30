c = 3e8;
fc = 850e6;
N = 498;
el = linspace(20,90,N);
az = zeros(1,N);
[cylrcs,cyl_az,cyl_el] = rcscylinder(1,1,20,c,fc);
cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylrcs,'Model','Swerling4');
data = cyltgt(ones(1,N),[az;el],true);
% data = chickenpox_dataset;
% data = [data{:}];

figure
plot(data)
xlabel("Elevation angle")
ylabel("RCS values")
title("Rcs vs elevation angle")
% Partition the training and test data. Train on the first 90% of the sequence and test on the last 10%.
numTimeStepsTrain = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

% Standardize Data
% For a better fit and to prevent the training from diverging, standardize the training data to have zero mean and unit variance. At prediction time, you must standardize the test data using the same parameters as the training data.
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

% Prepare Predictors and Responses
% To forecast the values of future time steps of a sequence, specify the responses to be the training sequences with values shifted by one time step. That is, at each time step of the input sequence, the LSTM network learns to predict the value of the next time step. The predictors are the training sequences without the final time step.
XTrain = dataTrainStandardized(1:end-1)
YTrain = dataTrainStandardized(2:end)
% 
% Define LSTM Network Architecture
% Create an LSTM regression network. Specify the LSTM layer to have 200 hidden units.
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
% Specify the training options. Set the solver to 'adam' and train for 250 epochs. To prevent the gradients from exploding, set the gradient threshold to 1. Specify the initial learn rate 0.005, and drop the learn rate after 125 epochs by multiplying by a factor of 0.2.
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Train LSTM Network
% Train the LSTM network with the specified training options by using trainNetwork.
net = trainNetwork(XTrain,YTrain,layers,options);

% Forecast Future Time Steps
% To forecast the values of multiple time steps in the future, use the predictAndUpdateState function to predict time steps one at a time and update the network state at each prediction. For each prediction, use the previous prediction as input to the function.
% Standardize the test data using the same parameters as the training data.
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
% To initialize the network state, first predict on the training data XTrain. Next, make the first prediction using the last time step of the training response YTrain(end). Loop over the remaining predictions and input the previous prediction to predictAndUpdateState.
% For large collections of data, long sequences, or large networks, predictions on the GPU are usually faster to compute than predictions on the CPU. Otherwise, predictions on the CPU are usually faster to compute. For single time step predictions, use the CPU. To use the CPU for prediction, set the 'ExecutionEnvironment' option of predictAndUpdateState to 'cpu'.
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
% Unstandardize the predictions using the parameters calculated earlier.
YPred = sig*YPred + mu;
%The training progress plot reports the root-mean-square error (RMSE) calculated from the standardized data. Calculate the RMSE from the unstandardized predictions.
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))
%Plot the training time series with the forecasted values.
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Elevation")
ylabel("RCS")
title("Prediction")
legend(["Observed" "Predicted"])
%Compare the forecasted values with the test data.
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("RCS")
title("Prediction")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Elevation angle")
ylabel("Error")
title("RMSE = " + rmse)

% Update Network State with Observed Values
% If you have access to the actual values of time steps between predictions, then you can update the network state with the observed values instead of the predicted values.
% First, initialize the network state. To make predictions on a new sequence, reset the network state using resetState. Resetting the network state prevents previous predictions from affecting the predictions on the new data. Reset the network state, and then initialize the network state by predicting on the training data.
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
%Predict on each time step. For each prediction, predict the next time step using the observed value of the previous time step. Set the 'ExecutionEnvironment' option of predictAndUpdateState to 'cpu'.
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end
%Unstandardize the predictions using the parameters calculated earlier.
YPred = sig*YPred + mu;
%Calculate the root-mean-square error (RMSE).
rmse = sqrt(mean((YPred-YTest).^2))
%Compare the forecasted values with the test data.
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("RCS")
title("Prediction with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Elevation angle")
ylabel("Error")
title("RMSE = " + rmse)

figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(data)
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
xlabel("Elevation")
ylabel("RCS")
title("Prediction")
legend(["Observed" "Predicted"]);

% xlabel("Elevation angle")
% ylabel("RCS values")
% title("Rcs vs elevation angle")
