clear all;
close all;

digits=10;
bins=32;
mode=0; %0 discrete  1 continuous
%-----load train image------
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
[pixels, image_num] = size(train_images);
train_bin_images = bin_image(train_images,bins);
%-----load test image------
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
test_bin_images = bin_image(test_images,bins);

switch(mode)
    case 0
        'discrete'
    %----prior---
        [prior,number_probrability] =discrete_prior(train_bin_images,train_labels,bins,digits);
    %----posterior---
        posterior = discrete_posterior(test_bin_images,digits,prior,number_probrability);
    %--------output posterior-------%
        outputfolder = 'output';
        output_posterior_file = sprintf('%s/posterior_discrete.txt',outputfolder);
        output_posterior(posterior,output_posterior_file,test_labels);
    %--------output image-------%
        output_prior_img(prior,outputfolder);
    
    case 1
        'continuous'
        [mean,var,number_probrability] =mean_var(train_labels,train_images,digits);
        posterior =cont_posterior(test_images,mean,var,number_probrability,digits);
    %--------output posterior-------%
        outputfolder = 'output2';
        output_posterior_file = sprintf('%s/posterior_continuous.txt',outputfolder);
        output_posterior(posterior,output_posterior_file,test_labels);
    %--------output image-------%
        output_img_cont(mean,outputfolder);
end




function output_posterior(posterior,filename,test_labels)
[digits,test_img_num] = size(posterior);
result = posterior./sum(posterior);
[M,I] = min(result);
rate=0;
for k=1:test_img_num   
  if(test_labels(k,1) ==mod(I(1,k),digits))
      rate=rate+1;
  end
end
match_rate = rate/test_img_num

post_file = fopen(filename,'w');
for i=1:test_img_num
    fprintf(post_file,'Posterior of image %d:\r\n',i);
    for k=1:digits
        fprintf(post_file,'%d: %010.9f\r\n',k,result(k,i));
    end
     fprintf(post_file,'Predict:%d Ans:%d\r\n',mod(I(1,i),digits),test_labels(i,1));
end
fclose(post_file);
end


function train_binary_images=bin_image(train_images,bins)
    divisor = 256/bins;
    train_binary_images = zeros(size(train_images));
    train_binary_images=floor(train_images/divisor) ; %32 bins
end

function [likelihood,number_probrability] =discrete_prior(train_binary_images,train_labels,bins,digits)

    [pixels, image_num] = size(train_binary_images);
  
pixel_hist =  0.00001*ones(pixels,bins,digits);  
label_count = 0.00001*ones(digits,1);
%total_histogram = zeros(bins,pixels);
for i = 1:image_num
    label = train_labels(i,1);
    if(label==0) 
        label=10;
    end
    label_count(label,1)= label_count(label,1)+1;
    for j = 1:pixels        
        pixel_value = train_binary_images(j,i) +1; % +1 to avoid value=zero
      %  total_histogram(pixel_value,j) =total_histogram(pixel_value,j)+1;
        pixel_hist(j,pixel_value,label)=pixel_hist(j,pixel_value,label)+1;       
    end
end

%total_histogram = total_histogram/image_num;
number_probrability = label_count/image_num;

for k=1:10
    likelihood(:,:,k) = pixel_hist(:,:,k)/label_count(k,1);
end
%prior = bsxfun(@rdivide, pixel_hist, label_count);
%prior = bsxfun(@ldivide, pixel_hist, label_count); 

end


function posterior = discrete_posterior(test_bin_images,digits,likelihood,number_probrability)
[pixels, test_img_num] = size(test_bin_images);
posterior = zeros(digits,test_img_num);
for i=1:test_img_num    
    for k=1:digits
        for j=1:pixels        
            value= test_bin_images(j,i);
            value_index = value+1;
            posterior(k,i)=posterior(k,i)+log(likelihood(j,value_index,k));            
        end        
        posterior(k,i)=posterior(k,i)+log(number_probrability(k,1));
    end    
end
end


function output_prior_img(likelihood,outputfolder)
[pixels,bins,digits]=size(likelihood);
width =sqrt(pixels);
output_img=zeros(width,width,digits);

for k=1:digits
    for j=1:pixels
        if(sum(likelihood(j,1:bins/2,k)) < sum(likelihood(j,bins/2+1:bins,k)))            
            output_img( mod(j,width),floor(j/width),k)=1;
        end
    end
end

%--------output image---------%

for k=1:digits
    fname = sprintf('%s/digit%d.jpeg', outputfolder,k);
    imwrite(output_img(:,:,k),fname,'JPEG');
end
%output_img(:,:,7)
end

function output_img_cont(mean,outputfolder)
[digits,pixels] = size(mean);
width =sqrt(pixels);
for k=1:digits
    output_img=zeros(width,width);
    for i=1:pixels
        if(mean(k,i)>=128)
            output_img(mod(i,width),floor(i/width))=1;
        end
    end
    fname = sprintf('%s/digit%d.jpeg', outputfolder,k);
    imwrite(output_img,fname,'JPEG');
end
end

%fname = sprintf('Gaussian_digit%d.jpeg', 1);
%imwrite(output_img(:,:,1),fname,'JPEG');

function [mean,var,number_probrability] =mean_var(train_labels,train_images,digits)
[pixels, image_num] = size(train_images);
sum_mat = zeros(digits,pixels);
label_count = 0.00001*ones(digits,1);

for i = 1:image_num
    label = train_labels(i,1);
    if(label==0) 
        label=10;
    end
    label_count(label,1)= label_count(label,1)+1;
    for j = 1:pixels            
        sum_mat(label,j) =sum_mat(label,j)+train_images(j,i);
    end
    
end
mean = sum_mat./ label_count;
number_probrability = label_count/image_num;

sum_var = zeros(digits,pixels);
for i = 1:image_num
    label = train_labels(i,1);
    if(label==0) 
        label=10;
    end
   % label_count(label,1)= label_count(label,1)+1;
    for j = 1:pixels       
        sum_var(label,j) =sum_var(label,j) +(train_images(j,i)-mean(label,j))^2;
    end    
end

var =  sum_var./label_count;
var(var==0)=0.000001;
end

function posterior =cont_posterior(test_images,mean,var,number_probrability,digits)
[pixels, test_img_num] = size(test_images);
posterior = zeros(digits,test_img_num);
for i = 1:test_img_num   
    for k=1:digits    
        for j = 1:pixels
           % likelihood = 2*pi*var(k,j)^(-0.5)*exp(-0.5*(test_images(j,i)-mean(k,j))^2/(var(k,j)));
           posterior(k,i)= posterior(k,i)-0.5*log(2*pi*var(k,j))-0.5*(test_images(j,i)-mean(k,j))^2/(var(k,j));
        end
         posterior(k,i)= posterior(k,i)+number_probrability(k);
    end    
end
end


