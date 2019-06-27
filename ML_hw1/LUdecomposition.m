A=[1 4 -3;-2 8 5; 3 4 7]
[L,U] = lu(A)
LI = inv(L)
UI = inv(U)
AI=UI*LI
inv(A)

U2=[1 4 -3; 0 16 -1; 0 0 15.5]
L2 = [1 0 0; -2 1 0; 3 -0.5 1]


L2*U2
X=inv(U2)
Y=inv(L2)
X*Y