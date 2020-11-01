%% Radar Target Classification Using Machine Learning and Deep Learning
% This example shows how to classify radar returns using feature extraction
% followed by a support vector machine (SVM) classifier. 
%
% In the second portion of the example, a deep learning workflow for the
% same data set is illustrated using a convolutional neural network and a
% Long Short-Term Memory (LSTM) recurrent neural network. Note that the 
% data set used in this example does not require advanced techniques but 
% the workflow is described because it can be used for more complex data 
% sets.     
%
% This example requires:
%
% * Phased Array System Toolbox
% * Wavelet Toolbox
% * Statistics and Machine Learning Toolbox
% * Deep Learning Toolbox

%% Introduction
% Target classification is an important function in modern radar systems.
% Because of the recent success of using machine learning techniques for
% classification, there is a lot of interest in applying similar techniques
% to classify radar returns. This example starts with a workflow where the
% SVM techniques are used to classify radar echoes from a cylinder and a
% cone. Although this example uses the synthesized I/Q samples, the
% workflow is applicable to real radar returns.
%
%% RCS Synthesis
% The next section shows how to create synthesized data to train the
% network. 
%
% The following code simulates the RCS pattern of a cylinder with a radius
% of 1 meter and a height of 10 meters. The operating frequency of the
% radar is 850 MHz.

c = 3e8;
fc = 850e6;
% [cylrcs,az,el] = rcscylinder(1,1,10,c,fc);
% helperTargetRCSPatternPlot(az,el,cylrcs);

%%
% The pattern can then be applied to a backscatter radar target to 
% simulate returns from different aspects angles.

% cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
%     'OperatingFrequency',fc,'AzimuthAngles',az,'ElevationAngles',el,'RCSPattern',cylrcs);

%%
% The following plot shows how to simulate 100 returns of the cylinder over
% time. It is assumed that the cylinder under goes a motion that causes
% small vibrations around bore sight, as a result, the aspect angle changes
% from one sample to the next.

% rng(2017);
% N = 100;
% az = 2*randn(1,N);                  % model vibration with +/- 2 degrees around boresight
% el = 2*randn(1,N);
% cylrtn = cyltgt(ones(1,N),[az;el]); % generate target echo 
% 
% clf
% plot(mag2db(abs(cylrtn)));
% xlabel('Time Index')
% ylabel('Target Return (dB)');
% title('Target Return for Cylinder');


%%
% The return of the cone can be generated similarly. To create the training
% set for the SVM classifier, the above process is repeated for 5
% arbitrarily selected cylinder radius. In addition, for each radius, 10
% motion profiles are simulated by varying the incident angle following 10
% randomly generated sinusoid curve around boresight. There are 701 samples
% in each motion profile, so there are 701x50 samples for each shape in
% total. Because of the long computation time, the training data is
% precomputed and loaded below.

% load('RCSClassificationReturnsTraining');

%%
% As an example, the next plot shows the return for one of the motion
% profiles from each shape. The plots show how the values change over time
% for both the incident azimuth angles and the target returns.

% clf;
% subplot(2,2,1); plot(cylinderAspectAngle(1,:)); ylim([-90 90]);
% title('Cylinder Aspect Angle vs. Time'); xlabel('Time Index'); ylabel('Aspect Angle (degrees)');
% subplot(2,2,3); plot(RCSReturns.Cylinder_1); ylim([-50 50]);
% title('Cylinder Return'); xlabel('Time Index'); ylabel('Target Return (dB)');
% subplot(2,2,2); plot(coneAspectAngle(1,:)); ylim([-90 90]);
% title('Cone Aspect Angle vs. Time'); xlabel('Time Index'); ylabel('Aspect Angle (degrees)');
% subplot(2,2,4); plot(RCSReturns.Cone_1); ylim([-50 50]);
% title('Cone Return'); xlabel('Time Index'); ylabel('Target Return (dB)');

%% Feature Extraction
% To improve the matching performance of learning algorithms, 
% the learning algorithms often work on extracted features rather than the
% original signal. The features make it easier for the classification
% algorithm to discriminate between returns from different targets. In
% addition, the features are often smaller in size compared to the original
% signal so it requires less computational resources to learn.
%
% There are a variety of ways to extract features for this type of data
% set. To obtain the right feature, it is often useful to take a look at a
% time-frequency view of data where the frequency due to motion is varying
% across radar pulses. The time-frequency signature of the signal can be
% derived by either Fourier transform (the spectrogram) or wavelet
% transforms. Specifically, this example uses wavelet packet representation
% of the signal. The following plots show the wavelet packet signatures for
% both the cone and the cylinder. These signatures provide some insight
% that the learning algorithms will be able to distinguish between the two.
% Specifically, there is separation between the frequency content over time
% between the two signatures.
cylinder = [];
cone =[];
N = 701;
for x = 1:500
      [conercs,cone_az,cone_el] = rcstruncone(0,5,10+x,c,fc);
    [cylrcs,cyl_az,cyl_el] = rcscylinder(1,1,20+x,c,fc);
    scatpos = [-0.5 -0.5 0.5 0.5;0.5 -0.5 0.5 -0.5;0 0 0 0];
    az = cone_az; naz=size(az,2);
    el = cone_el; nel=size(el,2);
    extcylrcs = zeros(nel,naz);
    extconercs = zeros(nel,naz);
    for m = 1:nel
        sv = steervec(scatpos,[az;el(m)*ones(1,naz)]);
        % sv is squared due to round trip in a monostatic scenario
        extcylrcs(m,:) = abs(sqrt(cylrcs(m,:)).*sum(sv.^2)).^2;
        extconercs(m,:) = abs(sqrt(conercs(m,:)).*sum(sv.^2)).^2;
    end
    cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',extcylrcs,'Model','Swerling4');
    conetgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cone_az,'ElevationAngles',cone_el,'RCSPattern',extconercs,'Model','Swerling4');
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
% cylinder = ones(size(cylinder));
% cones = ones(size(cone));
cylinder=awgn(cylinder,-30,'measured');
cone=awgn(cone,-30,'measured');
cylinder=tempcylinder(:,1:50);
cone=tempcone(:,1:50);
cyltesttable=array2table(cylinder);
conetesttable=array2table(cone);
RCSReturns=[cyltable conetable];
RCSTest=[cyltesttable conetesttable];
% levels = 3;
% figure;
% [wpt,~,F] = modwpt(RCSReturns{:,1},'fk6',levels,'TimeAlign',true);
% % clf;
% contour(1:size(wpt,2),F,abs(wpt).^2); grid on;
% xlabel('Time Index'); ylabel('Cycles per sample'); title('Wavelet Packet for Cylinder Return');
% 
% %%
% [wpt,~,F,E,RE] = modwpt(RCSReturns{:,51},'fk6',levels,'TimeAlign',true);
% % clf;
% figure;
% contour(1:size(wpt,2),F,abs(wpt).^2); grid on;
% xlabel('Time Index'); ylabel('Cycles per sample'); title('Wavelet Packet for Cone Return');

%%
% The apparent frequency separation between the cylinder and cone returns
% suggests using a frequency-domain measure to classify the signals. This
% example uses the maximal overlap discrete wavelet packet transform
% (MODWPT) to compare relative subband energies. The MODWPT at level $J$
% partitions the signal energy into $2^J$ equal-width subbands and does
% this in a way that the total signal energy is preserved.  To see this,
% you can do the following.

% T = array2table([F E RE*100],'VariableNames',{'CenterFrequency','Energy','PercentEnergy'});
% disp(T)

%%
% The table shows the subband center frequencies in cycles/sample, the
% energy in those subbands, and the percentage of the total energy in each
% subband. Note that MODWPT preserves the signal energy, which is an
% important property that is very difficult to achieve with conventional
% bandpass filtering. Specifically, you have a signal decomposition into
% subbands, which mimics an orthogonal transform. In signal classification
% problems where there is frequency separation between classes, this
% property significantly improves the ability of a classifier to accurately
% distinguish the classes.

%% 
% Using wavelet transform, the extracted features consists of 8 predictors
% per target return. Comparing to the original time domain signal of 701
% points, it is a significant reduction in the data. The number of levels
% for the wavelet transform can be tuned to improve performance of the
% classification algorithms.
%
% trainingData = varfun(@(x)helperWPTFeatureExtraction(x,'fk6',levels),RCSReturns);
% trainingData = array2table(table2array(trainingData)');
% % 50 cylinders followed by 50 cones
% shapeTypes = categorical({'Cylinder';'Cones'});
% trainingData.Type = shapeTypes([zeros(50,1); ones(50,1)]+1);

%% Model Training
% The Classification Learner app can be used to train the classifier. Once
% the training data is loaded, it can help apply different learning
% algorithms against the data set and report back the classification
% accuracy. The following picture is a snapshot of the app.
%
% <<../ClassificationLearnerInterface.png>>
%
% Based on the result, this example uses the SVM technique and then
% generates the corresponding training algorithm, |helperTrainClassifier|,
% from the app. The output of the training algorithm is a configured
% classifier, |trainedClassifier|, ready to perform the classification.

% [trainedClassifier, validationAccuracy] = helperTrainClassifier(trainingData);

%% Target Classification
% Once the model is ready, the network can process the received target
% return and perform classification using |predictFcn| method. The next
% section of the example creates a test data set using the approach similar
% to creating training data. This data is passed through the derived
% classifier to see if it can correctly classify the two shapes. The test
% data contains 25 cylinder returns and 25 cone returns. These cylinders
% and cones consist of 5 sizes for each shape and 5 motion profiles for
% each size. The generation process is the same as the training data, but
% with the specific values are slightly different due to randomness of the
% size values and incident angle values. The total number of samples for
% each shape is 701x25.

% load('RCSClassificationReturnsTest');
% 
% testData = varfun(@(x)helperWPTFeatureExtraction(x,'fk6',levels),RCSReturnsTest);
% testData = array2table(table2array(testData)');
% testResponses = shapeTypes([zeros(25,1); ones(25,1)]+1); % 25 cylinders followed by 25 cones
% 
% testPredictions = trainedClassifier.predictFcn(testData);
% cmat = confusionmat(testResponses, testPredictions)
% 
% %%
% % From the confusion matrix, it can be computed that the overall accuracy
% % is about 82%.
% 
% classacc = sum(diag(cmat))/sum(cmat(:))

%%
% It is possible to improve the performance of the classifier by increasing
% the quality and quantity of the training data. In addition, the feature
% extraction process can be improved to further distinguish characteristics
% of each target within the classification algorithm. Note that different
% features may have different optimal classification algorithms.

%%
% For more complex data sets, a deep learning workflow using a convolutional 
% neural network and a Long Short-Term Memory (LSTM) recurrent neural 
% network will also improve performance. For simplicity, the same data set
% will be used to demonstrate these workflows as well. 
%
%clear all;
%close all;
%load RCSClassificationReturnsTraining;
%load RCSClassificationReturnsTest;
% %% Transfer Learning
% % AlexNet is a deep CNN designed by Alex Krizhevsky and used in the ImageNet
% % Large Scale Visual Recognition Challenge (ILSVRC). Accordingly, AlexNet
% % has been trained to recognize images in 1,000 classes. In this example,
% % we reuse the pre-trained AlexNet to classify radar returns belonging to
% % one of two classes. To use AlexNet, you must install the Deep Learning
% % Toolbox&trade; Model _for AlexNet Network_ support package. To do this
% % use the MATLAB&trade; Add-On Explorer. If you have successfully installed
% % AlexNet, you can execute the following code to load the network and
% % display the network layers.
% anet = alexnet;
% anet.Layers
% %%
% % You see that AlexNet consists of 25 layers. Like all DCNNs, AlexNet
% % cascades convolutional operators followed by nonlinearities and pooling,
% % or averaging. AlexNet expects an image input of size 227-by-227-by-3,
% % which you can see with the following code.
% anet.Layers(1)
% %%
% % Additionally, AlexNet is configured to recognized 1,000 different
% % classes, which you can see with the following code.
% anet.Layers(23)
% %% 
% % In order to use AlexNet on our binary classification problem, we must
% % change the fully connected and classification layers.
% layers = anet.Layers;
% layers(23) = fullyConnectedLayer(2); 
% layers(25) = classificationLayer;
% %% Continuous Wavelet Transform
% % AlexNet is designed to discriminate differences in images and classify
% % the results. Therefore, in order to use AlexNet to classify radar
% % returns, we must transform the 1-D radar return time series into an
% % image. A common way to do this is to use a time-frequency representation
% % (TFR). There are a number of choices for a time-frequency representation
% % of a signal and which one is most appropriate depends on the signal
% % characteristics. To determine which TFR may be appropriate for this
% % problem, randomly choose and plot a few radar returns from each class.
%   rng default;
%   idxCylinder = randperm(50,2);
%   idxCone = randperm(50,2)+50;
%%
% % It is evident that the radar returns previously shown are characterized 
% % by slowing varying changes punctuated by large transient decreases as 
% % described earlier. A wavelet transform is ideally suited to sparsely
% % representing such signals. Wavelets shrink to localize transient
% % phenomena with high temporal resolution and stretch to capture slowly
% % varying signal structure. Obtain and plot the continuous wavelet
% % transform of one of the cylinder returns.
% cwt(RCSReturns{:,idxCylinder(1)},'VoicesPerOctave',8)

%%
% % The CWT simultaneously captures both the slowly varying (low frequency)
% % fluctuations and the transient phenomena. Contrast the CWT of the cylinder
% % return with one from a cone target.
% cwt(RCSReturns{:,idxCone(2)},'VoicesPerOctave',8);
%%
% Because of the apparent importance of the transients in determining
% whether the target return originates from a cylinder or cone target, we
% select the CWT as the ideal TFR to use.
% After obtaining the CWT for each target return, we make images
% from the CWT of each radar return. These images are resized to be
% compatible with AlexNet's input layer and we leverage AlexNet to classify
% the resulting images.
% %% Image Preparation
% % The helper function, |helpergenWaveletTFImg|, obtains the CWT for each
% % radar return, reshapes the CWT to be compatible with AlexNet, and writes
% % the CWT as a jpeg file. To run |helpergenWaveletTFImg|, choose a
% % |parentDir| where you have write permission. This example uses |tempdir|,
% % but you may use any folder on your machine where you have write
% % permission. The helper function creates |Training| and |Test| set folders
% % under |parentDir| as well as creating |Cylinder| and |Cone| subfolders
% % under both |Training| and |Test|. These folders are populated with jpeg
% % images to be used as inputs to AlexNet.
% % 
% parentDir = tempdir;
% helpergenWaveletTFImg(parentDir,RCSReturns,RCSReturnsTest)
% %%
% % Now use |imageDataStore| to manage file access from the folders in order
% % to train AlexNet. Create datastores for both the training and test data.
% trainingData= imageDatastore(fullfile(parentDir,'Training'), 'IncludeSubfolders', true,...
%     'LabelSource', 'foldernames');
% testData = imageDatastore(fullfile(parentDir,'Test'),'IncludeSubfolders',true,...
%     'LabelSource','foldernames');
% % %%
% % % Set the options for re-training AlexNet. Set the initial learn rate to
% % % 10^(-4), set the maximum number of epochs to 15, and the minibatch size
% % % to 10. Use stochastic gradient descent with momentum.
% % ilr = 1e-4;
% % mxEpochs = 15;
% % mbSize =10;
% % opts = trainingOptions('sgdm', 'InitialLearnRate', ilr, ...
% %     'MaxEpochs',mxEpochs , 'MiniBatchSize',mbSize, ...
% %     'Plots', 'training-progress','ExecutionEnvironment','cpu');
%%
% Train the network. If you have a compatible GPU, |trainNetwork|
% automatically utilizes the GPU and training should complete in less than
% one minute. If you do not have a compatible GPU, |trainNetwork| utilizes
% the CPU and training should take around five minutes. Training times do
% vary based on a number of factors. In this case, the training takes place
% on a cpu by setting the |ExecutionEnvironment| parameter to |cpu|.

% CWTnet = trainNetwork(trainingData,layers,opts);
%%
% Use the trained network to predict target returns in the held-out test
% set.
% predictedLabels = classify(CWTnet,testData);
% accuracy = sum(predictedLabels == testData.Labels)/50*100
%%
% % Plot the confusion chart along with the precision and recall. In this
% % case, 100% of the test samples are classified correctly.
% figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
% ccDCNN = confusionchart(testData.Labels,predictedLabels);
% ccDCNN.Title = 'Confusion Chart';
% ccDCNN.ColumnSummary = 'column-normalized';
% ccDCNN.RowSummary = 'row-normalized';

%% LSTM
% In the final section of this example, an LSTM workflow is described.
% First the LSTM layers are defined:

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
%trainData
% csvwrite('train.csv',trainData);
% csvwrite('test.csv',testData);
RNNnet = trainNetwork(trainData,trainLabels,LSTMlayers,options);
%% 
% The accuracy for this system is also plotted.

predictedLabels = classify(RNNnet,testData,'ExecutionEnvironment','cpu');
accuracy = sum(predictedLabels == testLabels)*100/size(testLabels,1)

%% Conclusion
% This example presents a workflow for performing radar target
% classification using machine learning techniques. Although this example
% used synthesized data to do training and testing, it can be easily
% extended to accommodate real radar returns.
% 
% In addition, the workflows for both a deep convolutional neural network with
% transfer learning and an LSTM network are described. This workflow can be 
% applied to more complex data sets. Because of the signal characteristics, 
% we chose the continuous wavelet transform as our ideal
% time-frequency representation. 

% Copyright 2019 The MathWorks, Inc.
