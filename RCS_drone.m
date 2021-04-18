clc;
clear all;
close all;
%% 

% filename='RCS_custom_drone.csv';
% data= readtable(filename);

% x= data.RCS_dB_;
% y = reshape(x,[],4);
% plot(y(:,1)); title('F450');grid on; axis tight;
% figure;
% plot(y(:,2));title('M100');grid on; axis tight;
% figure;
% plot(y(:,3));title('P4P');grid on; axis tight;
% figure;
% plot(y(:,4));title('Walkera');grid on; axis tight;
% 
% figure;
% plot(y(:,1));hold on;plot(y(:,2));plot(y(:,3));plot(y(:,4));title('Combined');grid on; axis tight;
% legend('F450','M100','P4P','Walkera');
%% 
filename='Drone_Data.csv';
data= readtable(filename);
data=data{:,:};
F450 = data(:,1:5);
M100=data(:,6:10);
P4P=data(:,11:15);
Walkera=data(:,16:20);

N=181;
NumObj=5;
NumTestObj=ceil(0.2*NumObj);
NumTrainObj=ceil(0.8*NumObj);
%%
% c = 3e8;
% fc = 850e6;
% cylinder = [];
% cone =[];
% sphere =[];
% disc =[];
% 
% N = 181;
% NumObj=4;
% NumTestObj=ceil(0.25*NumObj);
% NumTrainObj=ceil(0.75*NumObj);
% for x = 1:NumObj %each x corresponds to different radius object
%     [coneRcs,cone_az,cone_el] = rcstruncone(0,1+(x/3),1+(x/2),c,fc);
%     [cylRcs,cyl_az,cyl_el] = rcscylinder(1+(x/4),1+(x/2),0.5,c,fc);
%     [discRcs,disc_az,disc_el] = rcscylinder(5+(x/3),2+(x/3),2,c,fc);
%     [sprRcs,spr_az,spr_el] = rcstruncone(2,3+(x/2),2+(x/3),c,fc);
%     
% %     [sprRcs,spr_az,spr_el] = rcssphere(1+(x/2),c,fc);
% %     [discRcs,disc_az,disc_el] = rcsdisc(1+(x/2),c,fc);
%     
%     cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
%     'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylRcs,'Model','Swerling4');
%     conetgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
%     'OperatingFrequency',fc,'AzimuthAngles',cone_az,'ElevationAngles',cone_el,'RCSPattern',coneRcs,'Model','Swerling4');
%     sprtgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
%     'OperatingFrequency',fc,'AzimuthAngles',spr_az,'ElevationAngles',spr_el,'RCSPattern',sprRcs,'Model','Swerling4');
%     disctgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
%     'OperatingFrequency',fc,'AzimuthAngles',disc_az,'ElevationAngles',disc_el,'RCSPattern',discRcs,'Model','Swerling4');
%     
%     %rng(2017);
% %     az = 180*rand(1,N); 
% %     el = 90*rand(1,N); 
% %     r = a + (b-a).*rand(N,1);
%     az = linspace(2,178,N); 
%     el = linspace(2,88,N); 
%     
%     cylSeries=cyltgt(ones(1,N),[az;el],true).';
%     coneSeries=conetgt(ones(1,N),[az;el],true).';
%     sprSeries=sprtgt(ones(1,N),[az;el],true).';
%     discSeries=disctgt(ones(1,N),[az;el],true).';
%     
%     cylSeries=pow2db(cylSeries);
%     coneSeries=pow2db(coneSeries);
%     sprSeries=pow2db(sprSeries);
%     discSeries=pow2db(discSeries);
%     
%     cylinder = [cylinder cylSeries];
%     cone = [cone coneSeries];
%     sphere = [sphere sprSeries];
%     disc = [disc discSeries];
% end
%% 

%load('rcs_generated_cylinder.mat');
%csvwrite('cylinder_2.csv',cylinder);
cylTable=array2table(F450);
coneTable=array2table(M100);
spheretable=array2table(P4P);
discTable=array2table(Walkera);

% cylinder=awgn(cylinder,30,'measured');
% cone=awgn(cone,30,'measured');
% sphere=awgn(sphere,30,'measured');
% disc=awgn(disc,30,'measured');

testcylinder=F450(:,1:NumTestObj);
testcone=M100(:,1:NumTestObj);
testsphere=P4P(:,1:NumTestObj);
testdisc=Walkera(:,1:NumTestObj);

traincylinder=F450(:,NumTestObj+1:end);
traincone=M100(:,NumTestObj+1:end);
trainsphere=P4P(:,NumTestObj+1:end);
traindisc=Walkera(:,NumTestObj+1:end);

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

trainLabels = repelem(categorical({'F450','M100','P4P','Walkera'}),[NumTrainObj NumTrainObj NumTrainObj NumTrainObj]);
trainLabels = trainLabels(:);
trainData = num2cell(table2array(RCSReturns)',2);
testData = num2cell(table2array(RCSTest)',2);
testLabels = repelem(categorical({'F450','M100','P4P','Walkera'}),[NumTestObj NumTestObj NumTestObj NumTestObj]);
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
