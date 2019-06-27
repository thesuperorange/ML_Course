clc;
close all;
lamda=10000;
n=3;

testfile = fopen('testfile.txt','r');
formatSpec = '%f%f';
inputFile = textscan(testfile,formatSpec,'Delimiter', ',');
fclose(testfile);

x=cell2mat(inputFile(1));
y=cell2mat(inputFile(2));

%[rowNum colNum] = size(x);
A=ones(size(x));
for i = 1:n-1
	A=[A x.^i];
end
AT = transpose(A);
AT*A
X= inv(AT*A +lamda*eye(n)) * (AT*y)
N = inv(AT*A ) * (AT*y)

subplot(2,1,1)
scatter(x,y); hold on;
fplot(poly2sym(flipud(X)));

subplot(2,1,2)
scatter(x,y); hold on;
fplot(poly2sym(flipud(N)));


E = (A*X -y).^2;
sum(E)





