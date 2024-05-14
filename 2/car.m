clc
clear
figure;
grid on;
a = [5 5 6 6 7 7 10];
b = [0 1 1 2 2 3 3];

plot(a,b);
hold on;

theta = 0;
sim('controller');   
plot(x,y);

theta = 45;
sim('controller');
plot(x,y);

theta = -45;
sim('controller');
plot(x,y);

hold off;