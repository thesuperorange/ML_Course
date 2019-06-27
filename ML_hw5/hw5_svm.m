clear all;clc;
train_data=csvread('X_train.csv');
train_label=csvread('Y_train.csv');
test_data=csvread('X_test.csv');
test_label=csvread('Y_test.csv');

model_time=zeros(4);
pred_train_time=zeros(4);
pred_test_time=zeros(4);

%linear
%c=1 95.08
%c=8 95.00
%c=0.5 train 99.86 test 95.52
%c=2  train 100 test 95
timer1=tic;
model = svmtrain(train_label,train_data,'-s 0 -t 0 -c 2 ');
model_time(1) = toc(timer1);
[predict_label, accuracy1_1, dec_values]=svmpredict(train_label,train_data,model);
pred_train_time(1) = toc(timer1);
[predict_label, accuracy1_2, dec_values]=svmpredict(test_label,test_data,model);
pred_test_time(1) = toc(timer1);
%polynomial
%c=1 
%c=8 76.72
%c=8 gamma=0.0625 97.48
%cost=0.5 gamma=0.0625  28.92
timer2=tic;
model2 = svmtrain(train_label,train_data,'-s 0 -t 1 -c 0.5 -g 0.0625');
model_time(2) = toc(timer2);
[predict_label, accuracy2_1, dec_values]=svmpredict(train_label,train_data,model2);
pred_train_time(2) = toc(timer2);
[predict_label, accuracy2_2, dec_values]=svmpredict(test_label,test_data,model2);
pred_test_time(2) = toc(timer2);
%RBF
% c=0.5, g=0.0625  rate=97.08
%c=1, g=0.0625, rate=97.86
%c=2, g=0.0625, rate=97.94
%c=4, g=0.0625, rate=97.94
%c=8, g=0.0625, rate=97.94
timer3=tic;
model3 = svmtrain(train_label,train_data,'-s 0 -t 2 -c 2 -g 0.0625');
model_time(3) = toc(timer3);
[predict_label, accuracy3_1, dec_values]=svmpredict(train_label,train_data,model3);
pred_train_time(3) = toc(timer3);
[predict_label, accuracy3_2, dec_values]=svmpredict(test_label,test_data,model3);
pred_test_time(3) = toc(timer3);



%user-defined linear+RBF
linear_kernel=train_data*train_data';
linear_kernel_test=test_data*train_data';
sigma=0.0625
rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);
K =  [ (1:numel(train_label))',linear_kernel +rbfKernel(train_data,train_data)];
K2 =  [ (1:numel(test_label))',linear_kernel_test+rbfKernel(test_data,train_data)];
timer4=tic;
model_precomputed = svmtrain(train_label, K, '-s 0 -t 4 ');
model_time(4) = toc(timer4);
[predict_label_P, accuracy_P_1, dec_values_P] = svmpredict(train_label, K, model_precomputed);
pred_train_time(4) = toc(timer4);
[predict_label_P, accuracy_P_2, dec_values_P] = svmpredict(test_label, K2, model_precomputed);
pred_test_time(4) = toc(timer4);

fprintf("1.Linear kernel accuracy of train:%g, test:%g\n",accuracy1_1(1),accuracy1_2(1));
fprintf("2.Polynomial kernel accuracy of train:%g, test:%g\n",accuracy2_1(1),accuracy2_2(1));
fprintf("3.RBF kernel accuracy of train:%g, test:%g\n",accuracy3_1(1),accuracy3_2(1));
fprintf("4.RBF +linear kernel accuracy of train:%g, test:%g\n",accuracy_P_1(1),accuracy_P_2(1));



