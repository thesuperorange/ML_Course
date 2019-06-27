clear all;close all; clc;
x=csvread('Plot_X.csv');
y=csvread('Plot_Y.csv');
[datapoint,dim]=size(x);
%linear
cost=1;
model1 = svmtrain(y,x,sprintf('-s 0 -t 0 -c %g ', cost)); 
[predict_label_linear, accuracy1, dec_values]=svmpredict(y,x,model1);
%99.5667

%poly
cost=8;
model2 = svmtrain(y,x,sprintf('-s 0 -t 1 -c %g ', cost)); 
[predict_label_poly, accuracy2, dec_values]=svmpredict(y,x,model2);
%99.3333

%RBF
cost=8;gamma=0.0625;
model3 = svmtrain(y,x,sprintf('-s 0 -t 2 -c %g -g %g', cost,gamma)); 
[predict_label_RBF, accuracy3, dec_values]=svmpredict(y,x,model3);
% cost=1  gamma=5  99.9
% cost=0.25 gamma=0.25 =>99.4667
% cost=8 gamma=0.0625 =>99.4667
% cost=1 gamma=2  99.6667 sv太多
% cost=1 gamma=1024 100% sv太多
% cost=1 gamma=100 99.9667% sv太多
% cost=2 gamma=0.125 99.5%

%user-defined linear+RBF
sigma=64; %99.8667
rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
linearKernel = x*transpose(x);
K =  [ (1:datapoint)' , rbfKernel(x,x)+linearKernel ];
model4 = svmtrain(y,K,sprintf('-s 0 -t 4 -h 0 ')); %Nu-SVR change -s if you want SVC
[predict_label_precompute, accuracy4, dec_values]=svmpredict(y,K,model4);


figure
plot_result(predict_label_linear,model1.sv_indices,x,1,sprintf('linear kernel: %g',accuracy1(1)));
plot_result(predict_label_poly,model2.sv_indices,x,2,sprintf('poly kernel: %g',accuracy2(1)));
plot_result(predict_label_RBF,model3.sv_indices,x,3,sprintf('RBF kernel: %g',accuracy3(1)));
plot_result(predict_label_precompute,model4.sv_indices,x,4,sprintf('linear+RBF kernel: %g',accuracy4(1)));
function user_defined_kernel()
    sigma=0.0625;
    rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
    linearKernel = x*transpose(x);
end
function plot_result(label,sv_idx,data_point,f,fig_title)
[N col] =size(data_point);
subplot(2,2,f);
for i=1:N   
    if(label(i)==0) 
        color='r.';
    elseif(label(i)==1) 
        color='b.';
    elseif(label(i)==2) 
        color='g.';
    end
    plot(data_point(i,1),data_point(i,2),color);
    hold on;
end

for(ii=1:length(sv_idx))
   % idx = sv(ii,1);
  
     plot(data_point(sv_idx(ii),1),data_point(sv_idx(ii),2),'k*');
      hold on;
end

title(fig_title);
end