clc;
clear all;
close all;
%%
filename ='Drone_data_large.csv';
data = readtable(filename);
data=data{:,:};
F450 = data(:,1);
M100=data(:,2);
P4P=data(:,3);
Walkera=data(:,4);

F450 = F450(1:1810);
M100 = M100([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);
P4P = P4P([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);
Walkera = Walkera([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);

F450 = reshape(F450, [], 10);
M100 = reshape(M100, [], 10);
P4P = reshape(P4P, [], 10);
Walkera = reshape(Walkera, [], 10);

N=181;
NumObj=10;
NumTestObj=ceil(0.3*NumObj);
NumTrainObj=ceil(0.7*NumObj);

%%
Ffeat=[];
Mfeat=[];
Pfeat=[];
Wfeat=[];
for i=1:NumObj
    sigf = std(F450(:,i));  muf = mean(F450(:,i)); 
    sigm = std(M100(:,i));  mum = mean(M100(:,i));
    sigp = std(P4P(:,i)); mup = mean(P4P(:,i));
    sigw = std(Walkera(:,i)); muw = mean(Walkera(:,i));
    fstd = (F450(:,i) - muf) / sigf; 
    mstd = (M100(:,i) - mum) / sigm;
    pstd = (P4P(:,i) - mup) / sigp;
    wstd = (Walkera(:,i) - muw) / sigw;
    emdF450=emd(fstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdM100=emd(mstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdP4P=emd(pstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdWalkera=emd(wstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    memdF450=mean(emdF450);sigemdF450=std(emdF450);maxemdF50=max(emdF450);minemdF450=min(emdF450);
    memdM100=mean(emdM100);sigemdM100=std(emdM100);maxemdM100=max(emdM100);minemdM100=min(emdM100);
    memdP4P=mean(emdP4P);sigemdP4P=std(emdP4P);maxemdP4P=max(emdP4P);minemdP4P=min(emdP4P);
    memdWalkera=mean(emdWalkera);sigemdWalkera=std(emdWalkera);maxemdWalkera=max(emdWalkera);minemdWalkera=min(emdWalkera);
    Ffeat =[Ffeat; [muf memdF450 sigf sigemdF450 max(F450(:,i)) maxemdF50 min(F450(:,i)) minemdF450]];
    Mfeat =[Mfeat; [mum memdM100 sigm sigemdM100 max(M100(:,i)) maxemdM100 min(M100(:,i)) minemdM100]];
    Pfeat =[Pfeat; [mup memdP4P sigp sigemdP4P max(P4P(:,i)) maxemdP4P min(P4P(:,i)) minemdP4P]];
    Wfeat =[Wfeat; [muw memdWalkera sigw sigemdWalkera max(Walkera(:,i)) maxemdWalkera min(Walkera(:,i)) minemdWalkera]];
end

%%

p=0.2*NumObj;

testcylinder=Ffeat(p:2:p+2*(NumTestObj-1),:);
testcone=Mfeat(p:2:p+2*(NumTestObj-1),:);
testsphere=Pfeat(p:2:p+2*(NumTestObj-1),:);
testdisc=Wfeat(p:2:p+2*(NumTestObj-1),:);

traincylinder=Ffeat; traincylinder(p:2:p+2*(NumTestObj-1),:)=[];
traincone=Mfeat; traincone(p:2:p+2*(NumTestObj-1),:)=[];
trainsphere=Pfeat; trainsphere(p:2:p+2*(NumTestObj-1),:)=[];
traindisc=Wfeat; traindisc(p:2:p+2*(NumTestObj-1),:)=[];

RCSReturns=[traincylinder;traincone;trainsphere;traindisc];
RCSTest=[testcylinder;testcone;testsphere;testdisc];
% RCSReturns=array2table(RCSReturns);
% RCSTest=array2table(RCSTest);
% RCSReturns.Properties.VariableNames = {'Mean' 'Std' 'Max' 'Min' 'Emd1Mean' 'Emd1Std' 'Emd1Max' 'Emd1Min' 'Emd2Mean' 'Emd2Std' 'Emd2Max' 'Emd2Min' 'Emd3Mean' 'Emd3Std' 'Emd3Max' 'Emd3Min' 'Emd4Mean' 'Emd4Std' 'Emd4Max' 'Emd4Min'};
% RCSTest.Properties.VariableNames = {'Mean' 'Std' 'Max' 'Min' 'Emd1Mean' 'Emd1Std' 'Emd1Max' 'Emd1Min' 'Emd2Mean' 'Emd2Std' 'Emd2Max' 'Emd2Min' 'Emd3Mean' 'Emd3Std' 'Emd3Max' 'Emd3Min' 'Emd4Mean' 'Emd4Std' 'Emd4Max' 'Emd4Min'};

trainLabels = repelem(categorical({'F450','M100','P4P','Walkera'}),[NumTrainObj NumTrainObj NumTrainObj NumTrainObj]).';
testLabels = repelem(categorical({'F450','M100','P4P','Walkera'}),[NumTestObj NumTestObj NumTestObj NumTestObj]).';
% trainLabels = cellstr(trainLabels);
% testLabels = cellstr(testLabels);
% t = templateSVM('Standardize',true,'KernelFunction','gaussian');
Mdl = fitcecoc(RCSReturns,trainLabels);
predictedLabelSVM = predict(Mdl,RCSTest);
figure;
ccDCNN = confusionchart(testLabels,predictedLabelSVM);
ccDCNN.Title = 'Confusion Chart SVM';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';

Md2 = fitctree(RCSReturns,trainLabels);
predictedLabelsTree = predict(Md2,RCSTest);
figure;
ccDCNN = confusionchart(testLabels,predictedLabelsTree);
ccDCNN.Title = 'Confusion Chart Decision Tree';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';

Md3 = fitcknn(RCSReturns,trainLabels);
predictedLabelsKNN = predict(Md3,RCSTest);
figure;
ccDCNN = confusionchart(testLabels,predictedLabelsKNN);
ccDCNN.Title = 'Confusion Chart KNN';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';

accuracySVM = sum(predictedLabelSVM == testLabels)*100/size(testLabels,1)
accuracyTree = sum(predictedLabelsTree == testLabels)*100/size(testLabels,1)
accuracyKNN = sum(predictedLabelsKNN == testLabels)*100/size(testLabels,1)
%% 
RCSReturnscell=num2cell(RCSReturns.',1).';
RCSTestcell=num2cell(RCSTest.',1).';

%%
inputSize=20;
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

trainLabels = repelem(categorical({'F450','M100','P4P','Walkera'}),[NumTrainObj NumTrainObj NumTrainObj NumTrainObj]);
trainLabels = trainLabels(:);
testLabels = repelem(categorical({'F450','M100','P4P','Walkera'}),[NumTestObj NumTestObj NumTestObj NumTestObj]);
testLabels = testLabels(:);
RNNnet = trainNetwork(RCSReturnscell,trainLabels,LSTMlayers,options);
predictedLabelsRNN = classify(RNNnet,RCSTestcell,'ExecutionEnvironment','cpu');
accuracyRNN = sum(predictedLabelsRNN == testLabels)*100/size(testLabels,1)

%% 

figure;
ccDCNN = confusionchart(testLabels,predictedLabelsRNN);
ccDCNN.Title = 'Confusion Chart RNN';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';
