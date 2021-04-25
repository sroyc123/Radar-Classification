clc;
clear all;
close all;
%%
filename ='All_Drone.csv';
data = readtable(filename);
data=data{:,:};
F450 = data(:,1);
M100=data(:,2);
P4P=data(:,3);
Walkera=data(:,4);
Heli=data(:,6);
Hexa=data(:,7);
Mavic=data(:,8);% 0 to 180
Parrot=data(:,9);% 0 to 180
Y600=data(:,10);

F450 = F450(1:32580/2);
newM100=[];
newP4P=[];
newWalkera=[];
Heli = Heli(1:32580/2);
Hexa = Hexa(1:32580/2);
Mavic = Mavic(1:32580/2);
Parrot = Parrot(1:32580/2);
newY600=[];

x=6;
for i=1:90
    newM100 = [newM100;M100(x:x+180)];
    newP4P = [newP4P;P4P(x:x+180)];
    newWalkera = [newWalkera;Walkera(x:x+180)];
    newY600 = [newY600; Y600(x:x+180)];
    x=x+191;
end

F450 = reshape(F450, [], 90);
M100 = reshape(newM100, [], 90);
P4P = reshape(newP4P, [], 90);
Walkera = reshape(newWalkera, [], 90);
Heli = reshape(Heli, [], 90);
Hexa = reshape(Hexa, [], 90);
Mavic = reshape(Mavic, [], 90);
Parrot = reshape(Parrot, [], 90);
Y600 = reshape(newY600,[], 90);
N=181;
NumObj=90;
NumTestObj=ceil(0.2*NumObj);
NumTrainObj=ceil(0.8*NumObj);
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
% filename='Drone_Data.csv';
% data= readtable(filename);
% data=data{:,:};
% F450 = data(:,1:5);
% M100=data(:,6:10);
% P4P=data(:,11:15);
% Walkera=data(:,16:20);
% 
% N=181;
% NumObj=5;
% NumTestObj=ceil(0.2*NumObj);
% NumTrainObj=ceil(0.8*NumObj);
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

p=0.2*NumObj;

testF450 = F450(:,p:2:p+2*(NumTestObj-1));
testM100 = M100(:,p:2:p+2*(NumTestObj-1));
testP4P = P4P(:,p:2:p+2*(NumTestObj-1));
testWalkera = Walkera(:,p:2:p+2*(NumTestObj-1));
testHeli = Heli(:,p:2:p+2*(NumTestObj-1));
testHexa = Hexa(:,p:2:p+2*(NumTestObj-1));
testMavic = Mavic(:,p:2:p+2*(NumTestObj-1));
testParrot = Parrot(:,p:2:p+2*(NumTestObj-1));
testY600 = Y600(:,p:2:p+2*(NumTestObj-1));

trainF450=F450; trainF450(:,p:2:p+2*(NumTestObj-1))=[];
trainM100=M100; trainM100(:,p:2:p+2*(NumTestObj-1))=[];
trainP4P=P4P; trainP4P(:,p:2:p+2*(NumTestObj-1))=[];
trainWalkera=Walkera; trainWalkera(:,p:2:p+2*(NumTestObj-1))=[];
trainHeli=Heli; trainHeli(:,p:2:p+2*(NumTestObj-1))=[];
trainHexa=Hexa; trainHexa(:,p:2:p+2*(NumTestObj-1))=[];
trainMavic=Mavic; trainMavic(:,p:2:p+2*(NumTestObj-1))=[];
trainParrot=Parrot; trainParrot(:,p:2:p+2*(NumTestObj-1))=[];
trainY600=Y600; trainY600(:,p:2:p+2*(NumTestObj-1))=[];

RCSReturns=[trainF450 trainM100 trainWalkera trainY600];
RCSTest=[testF450 testM100 testWalkera testY600];

RCSReturnscell=num2cell(RCSReturns.',2);
RCSTestcell=num2cell(RCSTest.',2);
%% 

inputSize=1;
numHiddenUnits=100;
numClasses=4;
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
trainLabels = repelem(categorical({'F450','M100','P4P','Y600'}),[NumTrainObj NumTrainObj NumTrainObj NumTrainObj]).';
testLabels = repelem(categorical({'F450','M100','P4P','Y600'}),[NumTestObj NumTestObj NumTestObj NumTestObj]).';
trainLabels = trainLabels(:);
testLabels = testLabels(:);
RNNnet = trainNetwork(RCSReturnscell,trainLabels,LSTMlayers,options);
%%
predictedLabelsRNN = classify(RNNnet,RCSTestcell,'ExecutionEnvironment','cpu');
accuracyRNN = sum(predictedLabelsRNN == testLabels)*100/size(testLabels,1)

figure;
ccDCNN = confusionchart(testLabels,predictedLabelsRNN);
ccDCNN.Title = 'Confusion Chart RNN Raw data';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';

