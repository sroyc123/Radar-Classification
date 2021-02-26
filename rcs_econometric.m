c = 3e8;
fc = 850e6;
% N = 500;
% el = linspace(20,90,N);
% newel = linspace(0,90,N);
% az = zeros(1,N);
% [cylrcs,cyl_az,cyl_el] = rcscylinder(1,1,20,c,fc);
% cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
%     'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylrcs);
% data = cyltgt(ones(1,N),[az;el]);
N = 500;
el = linspace(20,90,N);
newel = linspace(0,90,N);
az = zeros(1,N);
[cylrcs,cyl_az,cyl_el] = rcscylinder(1,1,20,c,fc);
cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylrcs,'Model','Swerling4');
data = cyltgt(ones(1,N),[az;el],true);

data=data';
numTimeStepsTrain = floor(0.5*numel(data));
numTimeStepsTest = floor(0.2*numel(data));
numTimeStepsResidual = floor(0.3*numel(data));
traindata= data(1:numTimeStepsTrain);
testActual= data(numTimeStepsTrain+1:numTimeStepsTrain+numTimeStepsTest);

figure
plot(newel,data);
hold on;
plot(newel(numTimeStepsTrain+1:numTimeStepsTrain+numTimeStepsTest),testActual);
xlabel("Elevation angle")
ylabel("RCS values")
title("Rcs vs Elevation angle")
axis tight;


%MA
sys = arima(0,0,5);
Md1 = estimate(sys,traindata);
yf = forecast(Md1,numTimeStepsTest,'Y0',traindata);
figure;
% newel = linspace(0,90,length(traindata)+length(yf));
plot(newel,data,'b');
hold on;
plot(newel(numTimeStepsTrain:numTimeStepsTrain+length(yf)),[traindata(end);yf],'r');
legend('measured','forecasted');
title("MA")
xlabel("Elevation angle")
ylabel("RCS values")

%AR
sys = arima(4,0,0);
Md1 = estimate(sys,traindata);
yf = forecast(Md1,numTimeStepsTest,'Y0',traindata);
figure;
plot(newel,data,'b');
hold on;
plot(newel(numTimeStepsTrain:numTimeStepsTrain+length(yf)),[traindata(end);yf],'r');
legend('measured','forecasted');
title("AR")
xlabel("Elevation angle")
ylabel("RCS values")


%ARMA
sys = arima(4,0,5);
Md1 = estimate(sys,traindata);
yf = forecast(Md1,numTimeStepsTest,'Y0',traindata);
figure;
plot(newel,data,'b');
hold on;
plot(newel(numTimeStepsTrain:numTimeStepsTrain+length(yf)),[traindata(end);yf],'r');
legend('measured','forecasted');
title("ARMA")
xlabel("Elevation angle")
ylabel("RCS values")

%ARIMA
%% 
sys = arima(2,1,8);
Md1 = estimate(sys,traindata);
yf = forecast(Md1,numTimeStepsTest,'Y0',traindata);
figure;
plot(newel,data,'b');
hold on;
plot(newel(numTimeStepsTrain:numTimeStepsTrain+length(yf)),[traindata(end);yf],'r');
legend('measured','forecasted');
title("ARIMA")
xlabel("Elevation angle")
ylabel("RCS values")
%% 

%SARIMA
sys = arima('Constant',NaN,'ARLags',1:4,'D',0,'MALags',1:2,'SARLags',[12,24,36,48],'Seasonality',12,'SMALags',12,'Distribution','Gaussian');
Md1 = estimate(sys,traindata);
yf = forecast(Md1,numTimeStepsTest,'Y0',traindata);
figure;
plot(newel,data,'b');
hold on;
plot(newel(numTimeStepsTrain:numTimeStepsTrain+length(yf)),[traindata(end);yf],'r');
legend('measured','forecasted');
title("SARIMA")
xlabel("Elevation angle")
ylabel("RCS values")
