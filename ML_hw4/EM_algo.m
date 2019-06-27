clc;clear all;
close all;

digits=10;
bins=2;
%initial


%-----load train image------
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');
[pixels, image_num] = size(train_images);
train_bin_images = bin_image(train_images,bins);

%initial
a=0.4;b=0.6; %random range between 0.4~0.6 to avoid small number overflow
mu=(b-a).*rand(digits,pixels) + a;  %p(j,k) if pixel=1, the prob of pixel k belongs to digit j
lambda=0.1*ones(digits,1);    %lambda(i) the prob of image i

image_num=1000;
z=zeros(digits,image_num);

%iteration
iter=0;
while(1)
    iter=iter+1;
    fprintf('iteration %d',iter);
%E step 
    for k=1:image_num 
        prod_class=zeros(digits,1);
 %sum_p=0;
        for i=1:digits       
            prod=1;        
            for j=1:pixels
                if(train_bin_images(j,k)==1)                
                    prod=prod*mu(i,j);
                elseif(train_bin_images(j,k)==0)
                    prod=prod*(1-mu(i,j));
                end            
            end        
        prod_class(i)=lambda(i)*prod;        
    end
    sum_p=sum(prod_class);
    for i=1:digits  
        z(i,k)=prod_class(i)/sum_p;          
    end
end


%M step
N = sum(z,2); %10
x_mean = zeros(digits,pixels);
for m=1:digits
    for j=1:pixels
        sum_zx=0;  
        for n=1:image_num
            sum_zx=sum_zx+z(m,n)*train_bin_images(j,n);
        end
        x_mean(m,j) = (1/N(m)) * sum_zx;
    end
    lambda(m)=N(m)/image_num;
end
norm(x_mean-mu)
if(norm(x_mean-mu))<0.0001
    break;
end

mu=x_mean;
end

%evaluation
[M,I]=max(z);
draw_img(I,train_bin_images,digits,image_num,'output');
draw_img(train_labels(1:image_num),train_bin_images,digits,image_num,'labeled');

C=confusionmat(train_labels(1:1000),I)

% set the max number as cluster label
[V,idx]=max(C);
mapping = idx(2:11) %output mapping index
stat_count = zeros(image_num,digits);
origin_label = zeros(image_num,digits);
for i=1:image_num
    mapping_idx = mapping(I(i))-1;
    if(mapping_idx==0) 
        mapping_idx=10;
    end
    stat_count(i,mapping_idx)=1;   
    origin_idx = train_labels(i) ;
     if(origin_idx==0) 
        origin_idx=10;
    end
    origin_label(i,origin_idx)=1;
end
error_count =0;
for i=1:digits
    fprintf('Confusion Matrix %d:\n',i);
    confusion = confusionmat(stat_count(:,i),origin_label(:,i))
    error_count=error_count+confusion(1,2)+confusion(2,1);
    if (norm(confusion(1,:))~=0)
        sensitivity = confusion(1,1)/sum(confusion(1,:));
    else 
        sensitivity=0;
    end
    if (norm(confusion(2,:))~=0)
        specificity = confusion(2,2)/sum(confusion(2,:));
    else
        specificity = 0;
    end
    fprintf('sensitivity:Successful predict cluster 1: %f\n',sensitivity);
    fprintf('specificity:Successful predict cluster 2: %f\n',specificity);
    fprintf('Total image used: %d\n', image_num);
    fprintf('Total iteration to converge: %d\n',iter);
    fprintf('Total error rate: %f\n',error_count/(image_num*digits));
end


function draw_img(index,train_bin_images,digits,image_num,outputfolder)
    [pixels,origin_size]=size(train_bin_images);
    output_img = zeros(pixels,digits);
    count=zeros(1,digits);
    for k=1:image_num
        for i=1:digits
            if(index(k)==0)
                idx=10;
            else
                idx=index(k);
            end
            if(idx==i)
                count(1,i)=count(1,i)+1;    
                output_img(:,i)=output_img(:,i)+train_bin_images(:,k);
            end
        end
    end
    mean_img = output_img./count
    mean_img(mean_img<0.5)=0
    mean_img(mean_img>=0.5)=1
    r_mean_img=reshape(mean_img,[28,28,10]);
    %--------output image---------%
    for k=1:digits
        fname = sprintf('%s/digit%d.jpeg', outputfolder,k);
        imwrite(r_mean_img(:,:,k),fname,'JPEG');
    end
end

function train_binary_images=bin_image(train_images,bins)
    divisor = 256/bins;
    train_binary_images = zeros(size(train_images));
    train_binary_images=floor(train_images/divisor) ; %32 bins
end
