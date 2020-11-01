c = physconst('Lightspeed');
fc = 4.5e9;
rada = 4;
radb = 4;
hgt = 10;
el = -90:90;
az = 5;
[rcscyl,azcyl,elcyl] = rcscylinder(rada,radb,hgt,c,fc,az,el);
[rcscone,azcone,elcone] = rcstruncone(0,4,5,c,fc,az,el);
rcspat= rcscyl + rcscone;
plot(elresp,pow2db(rcspat));
% hold on;
% plot(elresp,pow2db(rcscyl));
% plot(elresp,pow2db(rcscone));
% xlabel('Elevation Angle (deg)')
ylabel('RCS (dB)')
title('RCS of cylinder+cone as Function of Elevation')
grid on
