clc; close all;
%read pgm data
P = 'att_faces/s*';
D = dir(fullfile(P,'*.pgm'));
C = cell(size(D));
mat = [];
for k = 1:numel(D)
    C{k} = imread(fullfile(D(k).folder,D(k).name));
    mat = [mat;reshape(cell2mat(C(k)),1,10304)];
end

%initial parameters
M=400;
height = 112;
width=92;
N=height*width;
mat=double(mat);
k=25;

%PCA
W=PCA_ml(mat,k);

% reshape to eigen face
eigenfaces=[];
for i=1:k
    c  = W(:,i);
    eigenfaces{i} = reshape(c,height,width);
end
%draw eigenfaces
z=[];
for i=0:4
z  = [z; eigenfaces{(i*5+1)}  eigenfaces{(i*5+2)}   eigenfaces{(i*5+3)} eigenfaces{(i*5+4)}     eigenfaces{(i*5+5)}  ];
end
figure(5),imshow(mat2gray(z),'Initialmagnification','fit');;title('eigenfaces')

%% reconstruct
n=10;
r = round(M*rand(n,1));
face_origin = mat(r,:);

X_rec = Vlarge*Vlarge'*A(:,1);

for i=1:n
    h=figure;
    face_r = reshape(X_rec(i,:),height,width);
    show_face = [reshape(face_origin(i,:),height,width) 255*ones(height,5) face_r];    
    imshow(mat2gray(show_face),'Initialmagnification','fit');
    title('origin v.s. reconstruction');
    saveas(h, sprintf('face_rec%d.jpg',r(i)));
    %imshow(mat2gray(show_face),'Initialmagnification','fit');
    
 end