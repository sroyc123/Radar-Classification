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
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;


emdData=emd(data,'Interpolation','pchip','Display',1);
