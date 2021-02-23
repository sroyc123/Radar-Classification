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

data=data';
newel = linspace(0,90,N);

%MA
sys = arima(0,0,20);
Md1 = estimate(sys,data);
yf = forecast(Md1,100,'Y0',data);
figure;
newel = linspace(0,90,length(data)+length(yf));
plot(newel(1:length(data)),data,'b',newel(length(data):length(data)+length(yf)),[data(end);yf],'r'), legend('measured','forecasted');
title("MA")
xlabel("Elevation angle")
ylabel("RCS values")

%AR
sys = arima(20,0,0);
Md1 = estimate(sys,data);
yf = forecast(Md1,100,'Y0',data);
figure; 
plot(newel(1:length(data)),data,'b',newel(length(data):length(data)+length(yf)),[data(end);yf],'r'), legend('measured','forecasted');
title("AR")
xlabel("Elevation angle")
ylabel("RCS values")


%ARMA
sys = arima(15,0,40);
Md1 = estimate(sys,data);
yf = forecast(Md1,100,'Y0',data);
figure;
plot(newel(1:length(data)),data,'b',newel(length(data):length(data)+length(yf)),[data(end);yf],'r'), legend('measured','forecasted');
title("ARMA")
xlabel("Elevation angle")
ylabel("RCS values")

%ARIMA
sys = arima(3,1,8);
Md1 = estimate(sys,data);
yf = forecast(Md1,100,'Y0',data);
figure;
plot(newel(1:length(data)),data,'b',newel(length(data):length(data)+length(yf)),[data(end);yf],'r'), legend('measured','forecasted');
title("ARIMA")
xlabel("Elevation angle")
ylabel("RCS values")

%SARIMA
sys = arima('Constant',NaN,'ARLags',1:4,'D',0,'MALags',1:2,'SARLags',[12,24,36,48],'Seasonality',12,'SMALags',12,'Distribution','Gaussian');
Md1 = estimate(sys,data);
yf = forecast(Md1,100,'Y0',data);
figure;
plot(newel(1:length(data)),data,'b',newel(length(data):length(data)+length(yf)),[data(end);yf],'r'), legend('measured','forecasted');
title("SARIMA")
xlabel("Elevation angle")
ylabel("RCS values")



