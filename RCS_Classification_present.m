c = 3e8;
fc = 850e6;
cylinder = [];
cone =[];
N = 701;
for x = 1:500
    [conercs,cone_az,cone_el] = rcstruncone(0,5,10+x,c,fc);
    [cylrcs,cyl_az,cyl_el] = rcscylinder(1,1,10+x,c,fc);
    cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylrcs,'Model','Swerling4');
    conetgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cone_az,'ElevationAngles',cone_el,'RCSPattern',conercs,'Model','Swerling4');
    %rng(2017);
    az = 180*rand(1,N); 
    el = 90*rand(1,N); 
    cylinder = [cylinder cyltgt(ones(1,N),[az;el],true).'];
    cone = [cone conetgt(ones(1,N),[az;el],true).'];
end
%load('rcs_generated_cylinder.mat');
%csvwrite('cylinder_2.csv',cylinder);
cyltable=array2table(cylinder);
conetable=array2table(cone);
tempcylinder=cylinder;
tempcone=cone;
cylinder=awgn(cylinder,30,'measured');
cone=awgn(cone,30,'measured');
cylinder=tempcylinder(:,1:50);
cone=tempcone(:,1:50);
cyltesttable=array2table(cylinder);
conetesttable=array2table(cone);
RCSReturns=[cyltable conetable];
RCSTest=[cyltesttable conetesttable];

LSTMlayers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 150, ...
    'InitialLearnRate', 0.01, ...
    'GradientThreshold', 1, ...
    'plots','training-progress', ...
    'Verbose',false,'ExecutionEnvironment','cpu');
trainLabels = repelem(categorical({'cylinder','cone'}),[500 500]);
trainLabels = trainLabels(:);
trainData = num2cell(table2array(RCSReturns)',2);
testData = num2cell(table2array(RCSTest)',2);
testLabels = repelem(categorical({'cylinder','cone'}),[50 50]);
testLabels = testLabels(:);
RNNnet = trainNetwork(trainData,trainLabels,LSTMlayers,options);


predictedLabels = classify(RNNnet,testData,'ExecutionEnvironment','cpu');
accuracy = sum(predictedLabels == testLabels)*100/size(testLabels,1)
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
ccDCNN = confusionchart(testLabels,predictedLabels);
ccDCNN.Title = 'Confusion Chart';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';
