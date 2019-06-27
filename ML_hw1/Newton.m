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
N = inv(AT*A ) * (AT*y)






