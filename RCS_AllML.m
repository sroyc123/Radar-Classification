clc;
clear all;
close all;
%%
% filename ='Drone_data_large.csv';
% data = readtable(filename);
% data=data{:,:};
% F450 = data(:,1);
% M100=data(:,2);
% P4P=data(:,3);
% Walkera=data(:,4);
% 
% F450 = F450(1:1810);
% M100 = M100([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);
% P4P = P4P([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);
% Walkera = Walkera([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);
% 
% F450 = reshape(F450, [], 10);
% M100 = reshape(M100, [], 10);
% P4P = reshape(P4P, [], 10);
% Walkera = reshape(Walkera, [], 10);
% 
% N=181;
% NumObj=10;
% NumTestObj=ceil(0.3*NumObj);
% NumTrainObj=ceil(0.7*NumObj);
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

% M100 = M100([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);
% P4P = P4P([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);
% Walkera = Walkera([6:186,197:377,388:568,579:759,770:950,961:1141,1152:1332,1343:1523,1534:1714,1725:1905]);

% 
% F450 = reshape(F450, [], 180);
% M100 = reshape(newM100, [], 180);
% P4P = reshape(newP4P, [], 180);
% Walkera = reshape(newWalkera, [], 180);
% Heli = reshape(Heli, [], 180);
% Hexa = reshape(Hexa, [], 180);
% Mavic = reshape(Mavic, [], 180);
% Parrot = reshape(Parrot, [], 180);
% Y600 = reshape(newY600,[], 180);

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
NumTestObj=ceil(0.3*NumObj);
NumTrainObj=ceil(0.7*NumObj);

%%
Ffeat=[];
Mfeat=[];
Pfeat=[];
Wfeat=[];
Hlfeat = [];
Hxfeat = [];
Mcfeat = [];
Ptfeat = [];
Yfeat=[];

for i=1:NumObj
    sigf = std(F450(:,i));  muf = mean(F450(:,i)); 
    sigm = std(M100(:,i));  mum = mean(M100(:,i));
    sigp = std(P4P(:,i)); mup = mean(P4P(:,i));
    sigw = std(Walkera(:,i)); muw = mean(Walkera(:,i));
    sigHl = std(Heli(:,i)); muHl = mean(Heli(:,i));
    sigHx = std(Hexa(:,i)); muHx = mean(Hexa(:,i));
    sigMc = std(Mavic(:,i)); muMc = mean(Mavic(:,i));
    sigPt = std(Parrot(:,i)); muPt = mean(Parrot(:,i));
    sigy = std(Y600(:,i)); muy = mean(Y600(:,i));
    
    fstd = (F450(:,i) - muf) / sigf; 
    mstd = (M100(:,i) - mum) / sigm;
    pstd = (P4P(:,i) - mup) / sigp;
    wstd = (Walkera(:,i) - muw) / sigw; 
    Hlstd = (Heli(:,i) - muHl) / sigHl;
    Hxstd = (Hexa(:,i) - muHx) / sigHx;
    Mcstd = (Mavic(:,i) - muMc) / sigMc;
    Ptstd = (Parrot(:,i) - muPt) / sigPt;
    ystd = (Y600(:,i) - muy) / sigy;
    
    emdF450=emd(fstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdM100=emd(mstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdP4P=emd(pstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdWalkera=emd(wstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdHeli=emd(Hlstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdHexa=emd(Hxstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdMavic=emd(Mcstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdParrot=emd(Ptstd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    emdY600=emd(ystd,'Interpolation','pchip','Display',1,"MaxNumIMF",4);
    
    memdF450=mean(emdF450);sigemdF450=std(emdF450);maxemdF50=max(emdF450);minemdF450=min(emdF450);
    memdM100=mean(emdM100);sigemdM100=std(emdM100);maxemdM100=max(emdM100);minemdM100=min(emdM100);
    memdP4P=mean(emdP4P);sigemdP4P=std(emdP4P);maxemdP4P=max(emdP4P);minemdP4P=min(emdP4P);
    memdWalkera=mean(emdWalkera);sigemdWalkera=std(emdWalkera);maxemdWalkera=max(emdWalkera);minemdWalkera=min(emdWalkera);
    memdHeli=mean(emdHeli);sigemdHeli=std(emdHeli);maxemdHeli=max(emdHeli);minemdHeli=min(emdHeli);
    memdHexa=mean(emdHexa);sigemdHexa=std(emdHexa);maxemdHexa=max(emdHexa);minemdHexa=min(emdHexa);
    memdMavic=mean(emdMavic);sigemdMavic=std(emdMavic);maxemdMavic=max(emdMavic);minemdMavic=min(emdMavic);
    memdParrot=mean(emdParrot);sigemdParrot=std(emdParrot);maxemdParrot=max(emdParrot);minemdParrot=min(emdParrot);
    memdY600=mean(emdY600);sigemdY600=std(emdY600);maxemdY600=max(emdY600);minemdY600=min(emdY600);
    
    Ffeat =[Ffeat; [muf memdF450 sigf sigemdF450 max(F450(:,i)) maxemdF50 min(F450(:,i)) minemdF450]];
    Mfeat =[Mfeat; [mum memdM100 sigm sigemdM100 max(M100(:,i)) maxemdM100 min(M100(:,i)) minemdM100]];
    Pfeat =[Pfeat; [mup memdP4P sigp sigemdP4P max(P4P(:,i)) maxemdP4P min(P4P(:,i)) minemdP4P]];
    Wfeat =[Wfeat; [muw memdWalkera sigw sigemdWalkera max(Walkera(:,i)) maxemdWalkera min(Walkera(:,i)) minemdWalkera]];
    Hlfeat =[Hlfeat; [muHl memdHeli sigHl sigemdHeli max(Heli(:,i)) maxemdHeli min(Heli(:,i)) minemdHeli]];
    Hxfeat =[Hxfeat; [muHx memdHexa sigHx sigemdHexa max(Hexa(:,i)) maxemdHexa min(Hexa(:,i)) minemdHexa]];
    Mcfeat =[Mcfeat; [muMc memdMavic sigMc sigemdMavic max(Mavic(:,i)) maxemdMavic min(Mavic(:,i)) minemdMavic]];
    Ptfeat =[Ptfeat; [muPt memdParrot sigPt sigemdParrot max(Parrot(:,i)) maxemdParrot min(Parrot(:,i)) minemdParrot]];
    Yfeat =[Yfeat; [muy memdY600 sigy sigemdY600 max(Y600(:,i)) maxemdY600 min(Y600(:,i)) minemdY600]];
end

%%

p=0.2*NumObj;

testF450 = Ffeat(p:2:p+2*(NumTestObj-1),:);
testM100 = Mfeat(p:2:p+2*(NumTestObj-1),:);
testP4P = Pfeat(p:2:p+2*(NumTestObj-1),:);
testWalkera = Wfeat(p:2:p+2*(NumTestObj-1),:);
testHeli = Hlfeat(p:2:p+2*(NumTestObj-1),:);
testHexa = Hxfeat(p:2:p+2*(NumTestObj-1),:);
testMavic = Mcfeat(p:2:p+2*(NumTestObj-1),:);
testParrot = Ptfeat(p:2:p+2*(NumTestObj-1),:);
testY600 = Yfeat(p:2:p+2*(NumTestObj-1),:);

trainF450=Ffeat; trainF450(p:2:p+2*(NumTestObj-1),:)=[];
trainM100=Mfeat; trainM100(p:2:p+2*(NumTestObj-1),:)=[];
trainP4P=Pfeat; trainP4P(p:2:p+2*(NumTestObj-1),:)=[];
trainWalkera=Wfeat; trainWalkera(p:2:p+2*(NumTestObj-1),:)=[];
trainHeli=Hlfeat; trainHeli(p:2:p+2*(NumTestObj-1),:)=[];
trainHexa=Hxfeat; trainHexa(p:2:p+2*(NumTestObj-1),:)=[];
trainMavic=Hxfeat; trainMavic(p:2:p+2*(NumTestObj-1),:)=[];
trainParrot=Hxfeat; trainParrot(p:2:p+2*(NumTestObj-1),:)=[];
trainY600=Yfeat; trainY600(p:2:p+2*(NumTestObj-1),:)=[];

RCSReturns=[trainF450;trainM100;trainP4P;trainWalkera; trainHexa; trainY600];
RCSTest=[testF450;testM100;testP4P;testWalkera; testHexa; testY600];
% RCSReturns=array2table(RCSReturns);
% RCSTest=array2table(RCSTest);
% RCSReturns.Properties.VariableNames = {'Mean' 'Std' 'Max' 'Min' 'Emd1Mean' 'Emd1Std' 'Emd1Max' 'Emd1Min' 'Emd2Mean' 'Emd2Std' 'Emd2Max' 'Emd2Min' 'Emd3Mean' 'Emd3Std' 'Emd3Max' 'Emd3Min' 'Emd4Mean' 'Emd4Std' 'Emd4Max' 'Emd4Min'};
% RCSTest.Properties.VariableNames = {'Mean' 'Std' 'Max' 'Min' 'Emd1Mean' 'Emd1Std' 'Emd1Max' 'Emd1Min' 'Emd2Mean' 'Emd2Std' 'Emd2Max' 'Emd2Min' 'Emd3Mean' 'Emd3Std' 'Emd3Max' 'Emd3Min' 'Emd4Mean' 'Emd4Std' 'Emd4Max' 'Emd4Min'};

trainLabels = repelem(categorical({'F450','M100','P4P','Walkera','Hexa','Y600'}),[NumTrainObj NumTrainObj NumTrainObj NumTrainObj NumTrainObj NumTrainObj]).';
testLabels = repelem(categorical({'F450','M100','P4P','Walkera','Hexa','Y600'}),[NumTestObj NumTestObj NumTestObj NumTestObj NumTestObj NumTestObj]).';
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
numClasses=6;
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

%trainLabels = repelem(categorical({'F450','M100','P4P','Walkera', 'Y600'}),[NumTrainObj NumTrainObj NumTrainObj NumTrainObj NumTrainObj]);
trainLabels = trainLabels(:);
%testLabels = repelem(categorical({'F450','M100','P4P','Walkera', 'Y600'}),[NumTestObj NumTestObj NumTestObj NumTestObj NumTestObj]);
testLabels = testLabels(:);
RNNnet = trainNetwork(RCSReturnscell,trainLabels,LSTMlayers,options);
predictedLabelsRNN = classify(RNNnet,RCSTestcell,'ExecutionEnvironment','cpu');
accuracyRNN = sum(predictedLabelsRNN == testLabels)*100/size(testLabels,1)

figure;
ccDCNN = confusionchart(testLabels,predictedLabelsRNN);
ccDCNN.Title = 'Confusion Chart RNN';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';
