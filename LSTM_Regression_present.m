c = 3e8;
fc = 850e6;
N = 498;
el = linspace(20,90,N);
newel = linspace(0,90,N);
az = zeros(1,N);
[cylrcs,cyl_az,cyl_el] = rcscylinder(1,1,20,c,fc);
cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylrcs,'Model','Swerling4');
data = cyltgt(ones(1,N),[az;el],true);
numTimeStepsTrain = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);
elTrain = newel(1:numTimeStepsTrain+1);
elTest = newel(numTimeStepsTrain+2:end);

figure
plot(newel,data)
xlabel("Elevation angle")
ylabel("RCS values")
title("Rcs vs Elevation angle")
axis tight;
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
YPred = sig*YPred + mu;
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))
figure
plot(newel,data)
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(elTest,YPred,'.-')
hold off
xlabel("Elevation angle")
ylabel("RCS")
title("Prediction")
legend(["Observed" "Predicted"])
axis tight;

figure
subplot(2,1,1)
plot(elTest,YTest)
hold on
plot(elTest,YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("RCS")
title("Prediction")
axis tight;

subplot(2,1,2)
stem(elTest,YPred - YTest)
xlabel("Elevation angle")
ylabel("Error")
title("RMSE = " + rmse)
axis tight;

% Update Network State with Observed Values
net = resetState(net);
net = predictAndUpdateState(net,XTrain);
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end
YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))
figure
subplot(2,1,1)
plot(elTest,YTest)
hold on
plot(elTest,YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("RCS")
title("Prediction with Updates")
axis tight;

subplot(2,1,2)
stem(elTest,YPred - YTest)
xlabel("Elevation angle")
ylabel("Error")
title("RMSE = " + rmse)
hold off
axis tight;
figure
idx = numTimeStepsTrain+1:(numTimeStepsTrain+numTimeStepsTest);
plot(newel,data)
hold on
plot(elTest,YPred,'.-');
xlabel("Elevation angle")
ylabel("RCS")
title("Prediction")
legend(["Observed" "Predicted" ]);
axis tight;
