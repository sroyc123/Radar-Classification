% clc;
% clear all;
% close all;
c = 3e8;
fc = 850e6;
cylinder = [];
cone =[];
az=[];
el=[];
radius=[];
answ=[];
N = 701;
for x = 1:500
    [conercs,cone_az,cone_el] = rcstruncone(0,5,10+x/10,c,fc);
    [cylrcs,cyl_az,cyl_el] = rcscylinder(1,1,20+x/10,c,fc);
    cyltgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cyl_az,'ElevationAngles',cyl_el,'RCSPattern',cylrcs,'Model','Swerling4');
    conetgt = phased.BackscatterRadarTarget('PropagationSpeed',c,...
    'OperatingFrequency',fc,'AzimuthAngles',cone_az,'ElevationAngles',cone_el,'RCSPattern',conercs,'Model','Swerling4');
    %rng(2017);
    naz = 180*rand(1,N); 
    nel = 90*rand(1,N); 
    cylinder = [cylinder ;cyltgt(ones(1,N),[naz;nel],true).'];
    cone = [cone ;conetgt(ones(1,N),[naz;nel],true).'];
    az= [az ;naz.'];
    el= [el ;nel.'];
    radius = [radius ;ones(N,1).*x];
end
answ = [answ cat(2,cylinder,az,el,radius,zeros(size(radius,1),1))];
answ = [answ ;cat(2,cone,az,el,radius,ones(size(radius,1),1))];
category = [zeros(1,size(radius,2)) ones(1,size(radius,2))];
T = array2table(answ,'VariableNames',{'rcs','az','el','radius','category'});
writetable(T,'data_close.csv');
