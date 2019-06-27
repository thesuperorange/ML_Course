clc;close all; clear all;

N=50;
mx1=1;my1=1;mx2=3;my2=3;
vx1=2;vy1=2;vx2=4;vy2=4;

%initial w
w=[1;1;1];
k=numel(w);
learning_rate=0.01;

X1=zeros(N,1);
X2=zeros(N,1);
Y1=zeros(N,1);
Y2=zeros(N,1);

th_n=0.5;th_g=0.5;
% figure
 for i=1:N
     
     X1(i)=univariate_gen(mx1,vx1);
     X2(i)=univariate_gen(mx2,vx2);
     Y1(i)=univariate_gen(my1,vy1);
     Y2(i)=univariate_gen(my2,vy2);
%     plot(X1(i),Y1(i),'ro');
%     hold on;
%     plot(X2(i),Y2(i),'b*');
%     hold on;
 end
data_point= [ X1 Y1];
data_point=[data_point;X2 Y2];
label=[zeros(N,1);ones(N,1)];
figure
plot_result(label,data_point,1,'Ground truth');

A=[label data_point];

w_new = Newton_train(A,w,label,learning_rate)
output=zeros(N*2,1);
result = zeros(N*2,1);
%use for loop because of hpf (big decimal)
for i=1:N*2
    output(i)=1/(1+exp(-(A(i,:)*w_new)));
end
result(output>=th_n)=1;
result(output<th_n)=0;
plot_result(result,data_point,2,'Newton Method');
C_n = confusionmat(label,result)
fprintf('sensitivity:Successful predict cluster 1: %f\n',C_n(1,1)/sum(C_n(1,:)))
fprintf('specificity:Successful predict cluster 2: %f\n',C_n(2,2)/sum(C_n(2,:)))


w_g = Gradient_train(A,w,label,learning_rate)
output_g=zeros(N*2,1);
result_g = zeros(N*2,1);
%use for loop because of hpf (big decimal)
for i=1:N*2
    output_g(i)=1/(1+exp(-(A(i,:)*w_g)));
end
result_g(output_g>=th_g)=1;
result_g(output_g<th_g)=0;
plot_result(result_g,data_point,3,'Gradient Descent');
C_g = confusionmat(label,result_g)
fprintf('sensitivity:Successful predict cluster 1: %f\n',C_g(1,1)/sum(C_g(1,:)))
fprintf('specificity:Successful predict cluster 2: %f\n',C_g(2,2)/sum(C_g(2,:)))

function plot_result(label,data_point,f,fig_title)
[N col] =size(data_point);
subplot(1,3,f);
for i=1:N   
    if(label(i)==0) 
        color='ro';
    elseif(label(i)==1) 
        color='b*';
    end
    plot(data_point(i,1),data_point(i,2),color);
    hold on;
end
title(fig_title);
end
function w_new=Newton_train(A,w,label,learning_rate)
    [N k]=size(A);
    AT=transpose(A);
    count=0;
    while(1)%for j=1:5000
        count=count+1;
%D(i,i) = e^-xw/(1+e^-xw)^2
        D=zeros(N,N);
%gradient=AT*(y-1/(1+e^(-xw)))
        gradient_temp = zeros(N,1);
        for i = 1:N
            x=A(i,:);  
            exponential_temp=exp(-(x*w));
%     if(isinf(exponential_temp))
%     fprintf('stop at %d',count);
%     break;
%     end
            D(i,i) = exponential_temp/((1+exponential_temp)^2);
            gradient_temp(i)=(1/(1+exponential_temp))-label(i);
        end
        Hession=AT*D*A;
        gradient=AT*gradient_temp;
        if(det(Hession) ==0) %singular
            fprintf('singular');
            w_new = w-learning_rate*gradient;
        else
            w_new = w-learning_rate*inv(Hession)*gradient;
        end
        w_new;
        %mean(abs(w-w_new))
        if(mean(abs(w-w_new))<0.0001)
            break;
        end
        w=w_new;
        count=count+1;
    end
    count
end
function w_new=Gradient_train(A,w,label,learning_rate)
    [N k]=size(A);
    AT=transpose(A);
    count=0;
    while(1)%for j=1:5000
        count=count+1;
%gradient=AT*(y-1/(1+e^(-xw)))
        gradient_temp = zeros(N,1);
        for i = 1:N
            x=A(i,:);
            exponential_temp=exp(-(x*w));
            gradient_temp(i)=1/(1+exponential_temp)-label(i);
        end
        gradient=AT*gradient_temp;
        w_new = w-learning_rate*gradient;
        %mean(abs(w-w_new))
        if(mean(abs(w-w_new))<0.0001)
            break;
        end
        w=w_new;
        count=count+1;
    end
    count
end

function AI=LU_inv(A)
    [L,U] = lu(A);
    LI = inv(L);
    UI = inv(U);
    AI=UI*LI;
end

function data_pt = univariate_gen(mean, var)
    std=sqrt(var);
    data_pt = std.*randn()+mean;
end