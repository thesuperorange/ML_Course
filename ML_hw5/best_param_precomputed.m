%# read dataset
%x=csvread('Plot_X.csv');
%y=csvread('Plot_Y.csv');
%[bestC,bestG]=grid(x,y)
x=csvread('X_train.csv');
y=csvread('Y_train.csv');
%[bestC,bestG]=grid(train_data,train_label)



%[dataClass, data] = libsvmread('./heart_scale');
%function [bestc,bestg]=grid(x,y)
[datapoint,dim] = size(x);
bestcv = 0;
linearKernel = x*transpose(x);
cmd = ['-v 5 -s 0 -t 4 -h 0'];
    
  for log2g = -4:1,
	sigma=2^log2g;	
    rbfKernel = @(X,Y) exp(-sigma .* pdist2(X,Y,'euclidean').^2);

	K =  [ (1:datapoint)' , rbfKernel(x,x)+linearKernel ];  
    cv = svmtrain(y, K, cmd);
    if (cv > bestcv),
      bestcv = cv;  bestg = 2^log2g;
    end
    fprintf('%g %g (best g=%g, rate=%g)\n',log2g, cv, bestg, bestcv);
  end
%end