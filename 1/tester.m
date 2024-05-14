clc
clear
mi =  18;     
T=0.01;
Kp = 200/150;
Ki = mi * Kp;
a = 1/mi;
a = a/2;
Ke =1.5;
K=Kp/(a*Ke);
Kd = a*Ke;
sim('Fuzzy');
output= ans.ScopeData.signals.values(1:1:end,2);
input = ans.ScopeData.signals.values(1:1:end,1);
t=ans.ScopeData.time;
figure;
plot(t,output);
hold on;
plot(t,input);



