close all; clc; clear all;
cov10=[0.121010411275987,0.0164696143446074,-0.134219761265593,-0.000190842768145422;0.0164696143446074,0.388082863883509,-0.0173906501421325,-0.266597423686020;-0.134219761265593,-0.0173906501421325,0.613268773792355,-0.0998156869938096;-0.000190842768145422,-0.266597423686020,-0.0998156869938096,0.764323924902738]
%xvec=[0.849739277065495,-0.814638769633098,-0.313693633993269,-0.203263076248893,-0.0315313511733431,0.322460741084608,0.322431019007013,-0.551911085622204,0.282741891256448,0.861577245982098]
%yvec=[7.41763079613806,0.0167700688395729,1.93213814293669,1.42622514721827,-0.556801117030948,2.51396877946037,2.61981350233676,-1.23112344131691,1.00731717667442,7.18222192185382]
mean_10=[1.43624936357598;2.82535168550312;1.37187064780139;1.63475354865171]

draw_varline(mean_10,cov10);

function draw_varline(mean,cov)
n=numel(mean);

x_data=-2:0.1:2

Y_high=zeros(size(x_data));
Y_low = zeros(size(x_data));
count=1;
for x=-2:0.1:2
    
%a=0;
X=zeros(1,4);
for i=1:n;
    X(i)=x.^(i-1);
    %X=[1 x x.^2 x.^3]a=a+mean_10(i)*x.^(i-1);
end
a=X*mean;
b=sqrt(X*cov*transpose(X))
Y_high(count)=X*mean+sqrt(X*cov*transpose(X)+1);
Y_low(count)=X*mean-sqrt(X*cov*transpose(X)+1);



%plot(x,a,'go')
%plot(x,Y_high,'b*')
%plot(x,Y_low,'r*')
%hold on
count=count+1;
end
figure
draw_smooth(x_data,Y_high,'red');
%hold on
draw_smooth(x_data,Y_low,'red');
end

function draw_smooth(x_data,y_data,color)
t = 1:numel(x_data);
xy = [x_data;y_data];
pp = spline(t,xy);
tInterp = linspace(1,numel(x_data));
xyInterp = ppval(pp, tInterp);

plot(x_data,y_data,'markersize',200')
hold on
plot(xyInterp(1,:),xyInterp(2,:),color)

end

