% This code is based on the paper entitled "ACCELERATED CONVERGENCE AND ENHANCED IMAGE
%DEBLURRING THROUGH NOVEL PRECONDITIONING TECHNIQUES FOR NONLINEAR SYSTEMS"

close all
clear all
clc
tic

nx  = 16; % Resize to reduce Problem


beta =0.01;  alpha = 1e-8;% Regulariztion parameters
%----------------Images-------------------------------------------------
 u_exact = double(imread('Cameraman.tif'));
%  u_exact = double(imread('moon.tif'));
%  u_exact = double(imread('kids.tif'));
u=imresize(u_exact,[nx nx]); %Resize Image

%-------------------Kernel-----------------------------------------------
 N = nx; h=0.5;
syms x y f
f=@(x,y) exp(-(x^2+y^2));

L=zeros(N,1); 
O1=ones(N,1);
X1=zeros(N,N);
Q=zeros(N^2,N^2);


M = cell(1, N);
for i = 1:N
        M{i} = zeros(N);
end


for k=1:N
for i=1:N
    D(i,1)=f((1-k)*h,(i-1)*h);
    T1 = spdiags(D(i,1)*O1, i-1, N,N);
    if i==1
        X1=T1;
    end
    M{k}=M{k}+T1+T1';
end
M{k}=M{k}-X1;



DD = repmat({M{k}},1,N-(k-1));
DDiag = blkdiag(DD{:});


Q(1:end-(k-1)*N,(k-1)*N+1:end) = Q(1:end-(k-1)*N,(k-1)*N+1:end) + DDiag;
Q((k-1)*N+1:end,1:end-(k-1)*N) = Q((k-1)*N+1:end,1:end-(k-1)*N)  + DDiag;

end
KK1=Q; 
 
%-------------------Blurry Image ----------------------------------------

z=conv2(KK1,u(:),'valid');
%Blur_psnr = ppsnr(z,u)

%-------------- Assemble the rhd of the system For Level I-----------------------

b1 =conv2(KK1',z','valid'); %k*kz
[b1r b1c]=size(b1'); 
n1 = nx^2;  m1 = 2*nx*(nx-1); 

%------------------------------MC-----------------------------------------
U1 = zeros(nx,nx); 
[D,C,AD] = computeDCA(U1,nx,m1,beta);

[B] = Bcomp(nx);
%---------------------Eigenvalues-------------------------------------------

 Q=[KK1'*KK1 -alpha*AD;
     zeros(nx^2)  eye(nx^2)];
[br bc]=size(B); 
 
OC =zeros([br, nx^2]);
 N=[-B OC;
     OC -B;
     OC OC];
 M=[OC' alpha*B' -alpha*B';
     -B'  OC'  OC'];
 
 OD =zeros(br, br);

 L=[D OD  OD;
     OD D  OD;
     -C OD D];
 
%--------------Coefficient Matrix-----------------------------------------------  
 A = [Q, M; 
      N, L];

[Ar Ac]=size(A);

%---------------------Given Matrix-----------------------------------------
b=zeros(Ar, 1);
b(1:b1r,1)=b1;

%---------------------Preconditioner---------------------------------------

%--------------------------------------------------------------------
 %---------------------GMRES Solver---------------------------------------
tol = 1e-8; % tolerance for FGMRES
restart = 5; % restart parameter for FGMRES

% Initialize FGMRES variables
x0 = zeros(Ar, 1);
maxit = 5; % maximum number of iterations for FGMRES

% FGMRES solver
tic;
[rm mm]=size(M);


[x, flag, relres, iter, resvec0] = gmres(A, b, restart, tol, maxit, [], [], x0);

 semilogy(resvec0)
 
%---------------------P1 Preconditioner---------------------------------------
OM=zeros(rm,mm);
P1 = [Q OM;
    N, L-N*inv(Q)*M];


[x, flag, relres, iter, resvec1] = gmres(A, b, restart, tol, maxit, P1, [], x0);

hold on
semilogy(resvec1)
 hold off
 

%----------------------------P2 Preconditioner------------------------------------------
% FGMRES solver
P2 = [Q 2.0001*M;
    N, L+N*inv(Q)*M];

[x, flag, relres, iter, resvec2] = gmres(A, b, restart, tol, maxit, P2, [], x0);
hold on
semilogy(resvec2)
 
legend('GMRES','P1','P2')
%title('Relative Residual Norms')

hold off
%--------------------------EigenValues ------------------------------------

figure;%A
eigenvalues_A = eig(full(A));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%P1^{-1}A
eigenvalues_A = eig(full(inv(P1)*A));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_1^{-1}A ');
xlabel('Real Part');
ylabel('Imaginary Part');

figure;%P2^{-1}A
eigenvalues_A = eig(full(inv(P2)*A));
plot((real(eigenvalues_A)),imag(eigenvalues_A),'o')
title('Eigenvalues of Matrix P_2^{-1}A ');
xlabel('Real Part');
ylabel('Imaginary Part');
%----------------Functions-------------------------------------------

function [D,C,A] = computeDCA(U,nx,m,beta);
h0=1/nx;
[X,Y] = meshgrid(h0/2:h0:1-h0/2);
% U = [20 26 100;5 10 30;25 30 40]  % give me U from previouw computations

nn = size(U,1);
UU = sparse(nn+2,nn+2);

% we are using reflection bounday conditions 
% another word, we are using normal boundary condition to be zero
UU(2:nn+1,2:nn+1) = U;
UU(1,:) = UU(2,:);
UU(nn+2,:) = UU(nn+1,:);
UU(:,1) = UU(:,2);
UU(:,nn+2) = UU(:,nn+1);
%------------------ Matrix D ------------------
Uxr = diff(U,1,2)/h0; % x-deriv at red points
xb = h0/2:h0:1-h0/2;   yr=xb;
yb = h0:h0:1-h0;       xr=yb;
[Xb,Yb]=meshgrid(xb,yb);
[Xr,Yr]=meshgrid(xr,yr);
Uxb = interp2(Xr,Yr,Uxr,Xb,Yb,'spline');
 
 
 Uyb = diff(U,1,1)/h0; % y-deriv at blue points
 Uyr = interp2(Xb,Yb,Uyb,Xr,Yr,'spline');
  
 
 Dr = sqrt( Uxr.^2 + Uyr.^2 + beta^2 );
 Db = sqrt( Uxb.^2 + Uyb.^2 + beta^2 );
 mm1 = size(Dr,1);
 
 Dvr = Dr(:);  Dvb = Db(:); Dv=[Dvr;Dvb];
 
 ddd = [ sparse(m,1) , Dv , sparse(m,1) ];
 D = spdiags(ddd,[-1 0 1],m,m);
 %-------------------------Matrix C----------------------------
 Wxr = diff(UU,3,2)/h0; % x-deriv at red points
 Wyb = diff(UU,3,1)/h0; % y-deriv at blue points
 
   
Wr = Wxr(1:mm1,:); 
Wb = Wyb(:,1:mm1); 
 

 Dwr = (Wr(:).*Uxr(:))./Dr(:);  Dwb = (Wb(:).*Uyb(:))./Db(:); Dw=[Dwr;Dwb];
 
 www = [ sparse(m,1) , Dw , sparse(m,1) ];
 C = spdiags(www,[-1 0 1],m,m);
 
 %-------------------- Matrix A -----------------------------
 
 E = zeros(nx,nx); 
 E(1,1)=1; E(nx,nx)=1;
 M=speye(nx,nx);

 A1 = kron(M,E);
 A2 = kron(E,M);
 A = 2*(A1 + A2)/(beta*h0);
end
 %-----------------------------------------------------------
 function [B] = Bcomp(nx)
e = ones(nx,1);
E = spdiags([0*e -1*e e], -1:1, nx, nx);
E1 =E(1:nx-1,:);
 
M1=eye(nx,nx);
B1=kron(E1,M1);
 
E2 = eye(nx);
M2 = spdiags([0*e -1*e e], -1:1, nx-1, nx);
B2 = kron(E2,M2);
 
B = [B1;B2];
 end

%--------------------------------------------------------------
function p = ppsnr(x,y)

% psnr - compute the Peack Signal to Noise Ratio, defined by :
%       PSNR(x,y) = 10*log10( max(max(x),max(y))^2 / |x-y|^2 ).
%
%   p = psnr(x,y);
%
%   Copyright (c) 2004 Gabriel Peyr

d = mean( mean( (x(:)-y(:)).^2 ) );
m1 = max( abs(x(:)) );
m2 = max( abs(y(:)) );
m = max(m1,m2);

p = 10*log10( m^2/d );
end
%------------------------------------------------------------------------
