x=[-2:0.1:2]
a=1
x1=x+1
y=1+2*x+3*x.^2+4*x.^3
y2=1.041410	+1.998979*x+	2.938378*x.^2+	3.947963	*x.^3
plot(x,y,x,y2)
hold on
x_vector=(0.880533)
y_vetor=(8.809685)
plot(x_vector,y_vetor,'o')


