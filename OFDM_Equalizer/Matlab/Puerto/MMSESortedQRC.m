%sorted QR en sus tres variantes
%% INPUT
%  tipo=0 realiza la descomposici?n Sorted QR utilizando Zero Forcing
%  tipo=1 realiza la descomposici?n Sorted QR utilizando MMSE
%  h --> Matriz de Canal
%  nstd --> Desviaci?n est?ndar del ruido
%  rec --> Vector recibido
%  psym --> Vector que contiene los simbolos piloto
%  ldata --> Numero de subpotadoras con datos
%  p --> Orden de datos y pilotos [DATA PILOTO] acorde al estandar V2V

%% OUTPUT
%  yp --> vector recibido ya con la cancelaci?n de las subportadoras piloto y
%  multiplicado por Q'
%  R1 --> Matriz triangular de dimensi?n ldata X ldata para realizar la
%  detecci?n de los datos contenidos en las ldata subportadoras
%  orden --> orden de detecci?n de las subportadoras con datos.

function [yp,R1,orden] = MMSESortedQRC(h,nstd,rec,ldata,tipo)

[m1 n1]=size(h);


%% inicializacion de variables
R1= zeros(48)+1i*zeros(48);
%R2= zeros(48,8)+1i*zeros(48,8);
orden = zeros(1,48);
r= zeros(48,48)+1i*zeros(48,48);
if(tipo==0)
  hm=zeros(48,48)+1i*zeros(48,48);
else
  hm=zeros(96,48)+1i*zeros(96,48);
end




% q=zeros(56)+1i*zeros(56);

yp = zeros(48,1)+1i*zeros(48,1);

%%



% se define que metodo utilizar
switch tipo
          case 0,
            hm=h;
          case 1,
            hm=[h;eye(n1)*nstd];
end
        
[m n] = size(hm);

% se inicializan la matriz Q R para el calculo de la descomposici?n 
% r=zeros(n,n);
q=hm;

p = zeros(1,48);
p = [1:1:48];

%calculo de las matrices Q y R
for k = 1:ldata,
 if(tipo==0)
    tras=m;
 else
    tras=m-n+k-1;
 end    
%  if(k<ldata)   
  ind=argmin3(q,k,ldata);
  auxiliar = q(1:tras,k);
  q(1:tras,k) = q(1:tras,ind);
  q(1:tras,ind) = auxiliar;
  auxiliar = r(:,k);
  r(:,k) = r(:,ind);
  r(:,ind) = auxiliar;
  
  auxiliar2=p(k);
  p(k)=p(ind);
  p(ind)=auxiliar2;
%  end
 
 r(k,k) = norm(q(:,k));
 q(:,k) = q(:,k)/r(k,k);
 for j = (k+1):n,
  r(k,j) = q(:,k)'*q(:,j);
  q(:,j) = q(:,j)-q(:,k)*r(k,j);
 end
 end 


switch tipo
          case 0,
            Q1=q;
            yp = Q1'*rec;
            R1=r;  
            orden=p;
          case 1,
            Q1=q(1:m1,:);
            yp = Q1'*rec;
            R1=r;
            orden=p;
end




function [indice] = argmin3(q,k,n)
norma = zeros(1,48);

smallest = inf;
indice=k;

% se calcula la norma de los conjuntos de vectores
for j=k:n,
 norma(j) =  sum((abs(q(:,j))).^2);
 if( norma(j) < smallest )
  smallest = norma(j);
  indice = j;
 end
end



% 
% 
% function [indice] = argmin3(q,k,n)
% norma = zeros(1,48)';
% biggest = 0;
% indice=k;
% 
% %se calcula la norma de los conjuntos de vectores
% for j=k:n,
%  norma(j) =  sum((abs(q(:,j))).^2);
%  if( norma(j) > biggest )
%   biggest = norma(j);