
output_img = zeros(784,10);
count=zeros(1,10);
for k=1:1000
for i=1:10
    if(I(k)==i)
        count(1,i)=count(1,i)+1;    
        output_img(:,i)=output_img(:,i)+train_bin_images(:,k);
    end
end
end
mean_img = output_img./count
mean_img(mean_img<0.5)=0
mean_img(mean_img>=0.5)=1
r_mean_img=reshape(mean_img,[28,28,10])
%--------output image---------%
 outputfolder='output';
 for k=1:digits
     fname = sprintf('%s/digit%d.jpeg', outputfolder,k);
     imwrite(r_mean_img(:,:,k),fname,'JPEG');
 end