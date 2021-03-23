c = 3e8;
fc = 850e6;
cylinder = {};
cone={};
N = 500;
NumObjects=100;
NumTestObj=ceil(0.1*NumObjects);
for x = 1:NumObjects %each x corresponds to different radius object
    [conercs,cone_az,cone_el] = rcstruncone(0,10,10+x,c,fc);
    [cylrcs,cyl_az,cyl_el] = rcscylinder(10,10,10+x,c,fc);
    cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylrcs,'Model','Swerling4');
    conetgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cone_az,'ElevationAngles',cone_el,'RCSPattern',conercs,'Model','Swerling4');
    %rng(2017);
    az=linspace(2,178,N);
    el=linspace(2,88,N);
    cylseries=cyltgt(ones(1,N),[az;el],true).';
    coneseries=conetgt(ones(1,N),[az;el],true).';
    cylseries=pow2db(cylseries);
    coneseries=pow2db(coneseries);
    mucyl = mean(cylseries); mucone = mean(coneseries);
    sigcyl = std(cylseries); sigcone = std(coneseries);
    cylstd = (cylseries - mucyl) / sigcyl; conestd = (coneseries - mucone) / sigcone;
    emdcyl=emd(cylstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdcone=emd(conestd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    cylinder(x,1)=num2cell(emdcyl.',[1 2]);
    cone(x,1)=num2cell(emdcone.',[1 2]);
end
%% 
%Splitting data for training
trainData={};
testData={};
for x = 1:NumTestObj
    trainData(x,1)=cylinder(x,1);
    testData(x,1)=cylinder(x,1);
end
for x = (NumTestObj+1):NumObjects
    trainData(x,1)=cylinder(x,1);
end

for x = 1:NumTestObj
    trainData(x+NumObjects,1)=cone(x,1);
    testData(x+NumTestObj,1)=cone(x,1);
end
for x = (NumTestObj+1):NumObjects
    trainData(x+NumObjects,1)=cylinder(x,1);
end

trainLabels = repelem(categorical({'cylinder','cone'}),[NumObjects NumObjects]);
trainLabels = trainLabels(:);
testLabels = repelem(categorical({'cylinder','cone'}),[NumTestObj NumTestObj]);
testLabels = testLabels(:);

%% 
%Defining neural network
inputSize=4;
numHiddenUnits=100;
numClasses=2;
LSTMlayers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 50, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'plots','training-progress', ...
    'Verbose',false,'ExecutionEnvironment','cpu');

RNNnet = trainNetwork(trainData,trainLabels,LSTMlayers,options);

% Plotting of accuracy.
predictedLabels = classify(RNNnet,testData,'ExecutionEnvironment','cpu');
accuracy = (sum(predictedLabels == testLabels)/size(testLabels,1))*100;
