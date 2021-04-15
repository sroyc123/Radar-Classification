c = 3e8;
fc = 850e6;

cylinder = [];
cone =[];
sphere =[];
disc =[];

N = 200;
NumObj=100;
NumTestObj=ceil(0.2*NumObj);
NumTrainObj=ceil(0.8*NumObj);
for x = 1:NumObj %each x corresponds to different radius object
    [coneRcs,cone_az,cone_el] = rcstruncone(0,10,10+(x/50),c,fc);
    [cylRcs,cyl_az,cyl_el] = rcscylinder(10,10,10+(x/50),c,fc);
    [sprRcs,spr_az,spr_el] = rcssphere(10+(x/50),c,fc);
    [discRcs,disc_az,disc_el] = rcsdisc(10+(x/50),c,fc);
    
    cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylRcs,'Model','Swerling4');
    conetgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cone_az,'ElevationAngles',cone_el,'RCSPattern',coneRcs,'Model','Swerling4');
    sprtgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',spr_az,'ElevationAngles',spr_el,'RCSPattern',sprRcs,'Model','Swerling4');
    disctgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',disc_az,'ElevationAngles',disc_el,'RCSPattern',discRcs,'Model','Swerling4');
    
    %rng(2017);
    az = 180*rand(1,N); 
    el = 90*rand(1,N); 
%     r = a + (b-a).*rand(N,1);
%     az = linspace(2,178,N); 
%     el = linspace(2,88,N); 
    
    cylSeries=cyltgt(ones(1,N),[az;el],true).';
    coneSeries=conetgt(ones(1,N),[az;el],true).';
    sprSeries=sprtgt(ones(1,N),[az;el],true).';
    discSeries=disctgt(ones(1,N),[az;el],true).';
    
    cylSeries=pow2db(cylSeries);
    coneSeries=pow2db(coneSeries);
    sprSeries=pow2db(sprSeries);
    discSeries=pow2db(discSeries);
    
    cylinder = [cylinder cylSeries];
    cone = [cone coneSeries];
    sphere = [sphere sprSeries];
    disc = [disc discSeries];
end
%% 

%load('rcs_generated_cylinder.mat');
%csvwrite('cylinder_2.csv',cylinder);
cylTable=array2table(cylinder);
coneTable=array2table(cone);
spheretable=array2table(sphere);
discTable=array2table(disc);

% cylinder=awgn(cylinder,30,'measured');
% cone=awgn(cone,30,'measured');
% sphere=awgn(sphere,30,'measured');
% disc=awgn(disc,30,'measured');

testcylinder=cylinder(:,1:NumTestObj);
testcone=cone(:,1:NumTestObj);
testsphere=sphere(:,1:NumTestObj);
testdisc=disc(:,1:NumTestObj);

traincylinder=cylinder(:,NumTestObj+1:end);
traincone=cone(:,NumTestObj+1:end);
trainsphere=sphere(:,NumTestObj+1:end);
traindisc=disc(:,NumTestObj+1:end);

cyltraintable=array2table(traincylinder);
conetraintable=array2table(traincone);
spheretraintable=array2table(trainsphere);
disctraintable=array2table(traindisc);

cyltesttable=array2table(testcylinder);
conetesttable=array2table(testcone);
spheretesttable=array2table(testsphere);
disctesttable=array2table(testdisc);

RCSReturns=[cyltraintable conetraintable spheretraintable disctraintable];
RCSTest=[cyltesttable conetesttable spheretesttable disctesttable];
%% 

LSTMlayers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(4)
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

trainLabels = repelem(categorical({'cylinder','cone','sphere','disc'}),[NumTrainObj NumTrainObj NumTrainObj NumTrainObj]);
trainLabels = trainLabels(:);
trainData = num2cell(table2array(RCSReturns)',2);
testData = num2cell(table2array(RCSTest)',2);
testLabels = repelem(categorical({'cylinder','cone','sphere','disc'}),[NumTestObj NumTestObj NumTestObj NumTestObj]);
testLabels = testLabels(:);
RNNnet = trainNetwork(trainData,trainLabels,LSTMlayers,options);

predictedLabels = classify(RNNnet,testData,'ExecutionEnvironment','cpu');
accuracy = sum(predictedLabels == testLabels)*100/size(testLabels,1)
%% 

figure;
ccDCNN = confusionchart(testLabels,predictedLabels);
ccDCNN.Title = 'Confusion Chart';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';
