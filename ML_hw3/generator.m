clc;close all; clear all;

% 1.a
var = 0.3;
mean=10;
data_pt = univariate_gen(mean,var)

% 1.b
w=[1,2,3,4];
[m n] = size(w);
a=1;
[x,y]=poly_gen(n,a,w)

%2
mean=3;
var=5;
sequential_estimator(mean,var);

%3
b=1;n=4;a=1;w=[1,2,3,4];
Baysian_LR(b,n,a,w);


function plotBaysian(a,b,n,w,x_vec,y_vec,last_mean,mean_10,mean_50,cov_10,cov_50,last_cov)
%plot
    x=[-2:0.1:2];
    y=pred_y(w,n);
    y2=pred_y(last_mean,n);
    y3=pred_y(mean_10,n);
    y4=pred_y(mean_50,n);

    figure
    subplot(2,2,1);
    plot(x,y)
    hold on
    draw_varline(transpose(w),eye(n),1,a);  %eye(n) useless
    title('Ground truth')
    
    subplot(2,2,2);
    plot(x,y2)
    hold on
    draw_varline(last_mean,last_cov,0,a);
    plot(x_vec,y_vec,'o')
    title('Predict Result')
    
    subplot(2,2,3);
    plot(x,y3)
    hold on
    draw_varline(mean_10,cov_10,0,a);
    plot(x_vec(1:10),y_vec(1:10),'o')
    title('After 10 incomes')
    
    subplot(2,2,4);
    plot(x,y4)
    hold on
    draw_varline(mean_50,cov_50,0,a);
    plot(x_vec(1:50),y_vec(1:50),'o')
    title('After 50 incomes')
end

function y=pred_y(w,n)
    x=[-2:0.1:2];
    y=0; 
    for i=1:n;
        y=y+w(i)*x.^(i-1);
    end
end

function Baysian_LR(b,n,a,w)
    filename = 'baysian_LR.txt';
    output_file = fopen(filename,'w');
    x_vec=[];
    y_vec=[];
    count=1;
    last_cov = zeros(n,n);  %last_covariance
    last_mean = zeros(n,1); 
    
    while(1)    
        [x,y]=poly_gen(n,a,w);
        x_vec=[x_vec x];
        y_vec=[y_vec y];
        fprintf(output_file,'[%d] Add data point (%f,%f):\r\n',count,x,y);    
        A=ones(size(x));
        for i = 1:n-1
            A=[A x.^i];
        end
        AT = transpose(A);    
        if(count==1)
            precision=a*AT*A+b*eye(n);
            covariance=inv(precision);
            mu=covariance*a*AT*y;
    
        else
            precision=a*AT*A+inv(last_cov);
            covariance=inv(precision);
            mu=covariance*(a*AT*y+inv(last_cov)*last_mean);
        end
        fprintf(output_file,'Posterior mean:\r\n');    
        fprintf(output_file,'%f\t',mu);
        fprintf(output_file,'\r\n');
        fprintf(output_file,'Posterior covariance:\r\n'); 
        fprintf(output_file,'%f\t',covariance);
        fprintf(output_file,'\r\n');
        pred_mean=A*mu;
        pred_var = 1/a+A*covariance*AT;
        fprintf(output_file,'Predictive distribution ~ N(%f,%f)\r\n',pred_mean,pred_var);
        
        if( (norm(mu - last_mean) - 0) < 0.00001)
             break;
        end
        if(count==10)
            mean_10=mu;
            cov_10=covariance;
        elseif(count==50)
            mean_50=mu;
            cov_50=covariance;
        end
        last_cov = covariance;
        last_mean = mu;        
        count=count+1;
    end
    fclose(output_file);
    plotBaysian(a,b,n,w,x_vec,y_vec,last_mean,mean_10,mean_50,cov_10,cov_50,last_cov);
    
end
function sequential_estimator(mean,var)
    filename = 'seq_estimator.txt';
    output_file = fopen(filename,'w');
    current_mean=0;
    current_var=0;
    last_mean =0;
    last_var =0;
    
    i=1;   
    while(1)
        data_pt = univariate_gen(mean,var);
        if(i==1)
            current_mean = data_pt/i;
            current_var = (data_pt-current_mean)^2/i;
    
        else
            current_mean =(current_mean*(i-1)+data_pt) / i;
            diff=current_mean^2-last_mean^2-2*(last_mean*(i-1))*(current_mean-last_mean) ; 
            current_var = (current_var*(i-1)+ diff+(data_pt-current_mean)^2) /i;
        end
        %if((current_mean==last_mean) & (current_var==last_var))
        if(abs(current_mean-last_mean)<0.0001 & abs(current_var-last_var)<0.0001)
            break;
        end
        last_mean = current_mean;
        last_var = current_var;
        fprintf(output_file,'[%d] data=%f, mean=%f, var=%f\r\n',i,data_pt,current_mean,current_var);
        i=i+1;
    end
    fclose(output_file);
end

function data_pt = univariate_gen(mean, var)
    std=sqrt(var);
    data_pt = std.*randn()+mean;
end

function [x,y]=poly_gen(n,a,w)
    upper=1;
    lower=-1;
    x = (upper-lower).*rand() + lower;    
    y=univariate_gen(0,a);
    for i=1:n;
        y=y+w(i)*x^(i-1);
    end
end

function draw_varline(mean,cov,tag,a)
    n=numel(mean);
    x_data=-2:0.1:2;
    Y_high=zeros(size(x_data));
    Y_low = zeros(size(x_data));
    
    count=1;
    for x=-2:0.1:2
        X=zeros(1,n);
        for i=1:n;
            X(i)=x.^(i-1);
        end
        if(tag==0)
            std=X*cov*transpose(X)+1/a;
        else
            std=a;
        end
        Y_high(count)=X*mean+std;
        Y_low(count)=X*mean-std;

        count=count+1;
    end
%figure
    draw_smooth(x_data,Y_high,'red');
    hold on
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

