 % Your data
 x=[-2:0.1:2]
y= 0.9291	+2.9077*x+	1.9515*x.^2+	1.5806	*x.^3
mean_10=[0.9291	2.9077	1.9515	1.5806]
n=4
figure
plot(x,y)
hold on
x = [    0.5691   -0.4515   -0.2049    0.7094    0.7207   -0.5530    0.5020    0.8376    0.6110   -0.9497];
y = [2.8310   -1.3527   -0.1627    5.1876    3.2674   -0.8204    3.7594    9.1147    3.8434   -1.0303];

newY_high=zeros(10)
newY_low=zeros(10)

for j=1:numel(x)
    fx=0; 
for i=1:n;
        fx=fx+mean_10(i)*x(j)^(i-1);
        
end
var = abs(fx-y(j))
newY_high(j)=y(i)+var
newY_low(j)=y(i)-var
end
[sortedX,idx]=sort(x)
sortedY_high=newY_high(idx)
sortedY_low=newY_low(idx)
% Cubic spline data interpolation
t = 1:numel(x);
xy = [sortedX;sortedY_high];
pp = spline(t,xy);
tInterp = linspace(1,numel(x));
xyInterp = ppval(pp, tInterp);
% Show the result

plot(sortedX,sortedY_high,'o:')
hold on
plot(xyInterp(1,:),xyInterp(2,:))
legend({'Original data','Spline interpolation'},'FontSize',12)