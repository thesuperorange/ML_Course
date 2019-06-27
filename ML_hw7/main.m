clc;close all;
X=csvread('mnist_X.csv');
label = csvread('mnist_label.csv');
   
[data_pt, dims] = size(X);

%PCA
W=PCA_ml(X,2);
X_PCA = X*W;
plot_cluster(X_PCA,label)
%LDA

W=LDA_ml(X,label,2);
X_LDA = X*W;
plot_cluster(X_LDA,label)

%draw
function plot_cluster(X,label)
[data_pt, dims] = size(X);
    colorstring = 'rbgyk';
    figure;
    for(i=1:data_pt)
        plot(X(i,1),X(i,2),colorstring(label(i)),'marker','.');
        hold on;
    end
end