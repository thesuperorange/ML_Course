close all; clc;

circle = csvread("circle.txt");
moon = csvread("moon.txt");
X=moon;
% 1. kmeans
%[cluster, center] = kmeans_ml(3,X,X,'kmeans_moon_3');  

% 2. kernel kmeans
%    X = circle;
%    theta=0.5;
%    W = rbf_kernel(X, X, 1 / (2 * theta^2 ));
%    kmeans_ml(2, W, X, 'kernelcircle2_');

%3. spectral kmeans
    X=circle;
    theta=0.1;
    W = rbf_kernel(X, X, 1 / (2 * theta^2 ));
    [class, L, U, value]=spectral_clustering(X, W, 4, 'spectral_moon_4',0);

%3.1 spectral draw in eigenspace
 % [class, L, U, value]=spectral_clustering(X, W, 4, 'spectral_circle_eigen_4',0);

% 4. DBSCAN
%[class,noise] = dbscan_ml(circle,5,0.1);
%[class2,noise2] = dbscan_ml(moon,5,0.1);

% %5. kmeans++
 %center = kmeanspp(X,3);
% [cluster, center] = kmeans_ml(3,X,X,'kmeanspp_moon_3_',center);  
