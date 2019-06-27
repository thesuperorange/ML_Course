clc; close all; clear all;
bernoulii(0,0);
bernoulii(10,1);

function bernoulii(a,b)
%a=0; b=0;
fprintf('initial a=%d, b=%d\n',a,b);
fid = fopen('testfile.txt');
tline = fgetl(fid);
line_count = 0;
while ischar(tline)
    line_count = line_count+1;
    fprintf('case %d: %s\n',line_count,tline);
    fprintf('Beta prior: a=%d b=%d\n',a,b);
    zero_count = 0;
    one_count = 0;
    N= length(tline);
    for i=1:N
        if(tline(i)=='1')
            one_count =one_count+1;
        elseif(tline(i)=='0')
            zero_count = zero_count+1;
        end
    end
    m=one_count;
    n=zero_count;
    a=m+a;
    b=n+b;
    p=m/N;
    fprintf('Likelihood: %f\n',nchoosek(N,m) * p^m * (1-p)^n);
    fprintf('Beta posterior: a=%d b=%d\n\n',a,b);
    tline = fgetl(fid);
end
fclose(fid);

end
