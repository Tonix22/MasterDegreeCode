function [s_est,nodos] = Near_ML(yp,R,conste,index)

% se realiza el preprocesamiento para deteccion Near_ML
  [row nt] = size(R);
  QRM = length(conste);
  
%% parametros de inici
  M=4;
  s_est = zeros(nt,1)+1i*zeros(nt,1);
  sest3 = zeros(nt,1)+1i*zeros(nt,1);
  sest2 = zeros(1,QRM);
  coder.extrinsic('cell2mat'); 
  coder.extrinsic('mat2cell'); 
%   coder.extrinsic('clear'); 

  acu=0;
  nodos=0;

  parent_yp = zeros(nt,QRM)+1i*zeros(nt,QRM);
  parent_yp2 = zeros(nt,M)+1i*zeros(nt,M);
  parent_yp2t = zeros(nt,M)+1i*zeros(nt,M);
  parent_node = zeros(1,2*QRM)+1i*zeros(1,2*QRM);
  parent_node2 = zeros(nt,2*M)+1i*zeros(nt,2*M);
  parent_node2t = zeros(1,2*M)+1i*zeros(1,2*M);
  vector = zeros(1,QRM*M);
  pos = zeros(1,QRM*M);
  x1 = zeros(48,2*M*QRM)+1i*zeros(48,2*M*QRM);
  distc = zeros(1,M*QRM)+1i*zeros(1,M*QRM);
  distc2=zeros(1,QRM);
  ordtotal = zeros(1,M*QRM);
  
%  ordtotal = zeros(1,(M*QRM)/2);
 % x1 = zeros(nt,QRM*M)+1i*zeros(nt,QRM*M);

% se realiza la detecci?n del nivel nt  
  a_est = yp(nt,1)/R(nt,nt);  
  sest2 = (abs(a_est - conste).^2);
  [dist ord]=sort(sest2);
  row=1;
  for p=1:QRM,
    parent_node(:,row:row+1) = [dist(p) conste(ord(p))];
    parent_yp(:,p)=yp;
    row=row+2;
  end

 
 indice=1;
 dmin=inf+1i*inf;
 skip=0;
 row=1;
for n=1:QRM,     
% se realiza la estimaci?n de los nt-1 niveles
  for k=nt-1:-1:1,
    if(k==nt-1)
       distp = parent_node(1,row);
       sest = parent_node(1,row+1);
       rm = parent_yp(:,n);
       rm = rm - sest*R(:,k+1);
       a_est = rm(k,1)/R(k,k);
       sest2 = (abs(a_est - conste).^2);
       distc2 = distp + sest2;  
       [dist2 ord2]=sort(distc2);
       col=1;
       for p=1:M,
        parent_node2(k+1,col:col+1) = parent_node(1,row:row+1);
        parent_node2(k,col:col+1)=[dist2(p) conste(ord2(p))]; 
        parent_yp2(:,p) = rm;
        col=col+2;
       end
       row=row+2;
    else
       acu=1; 
       col=1;
       for p=1:M,
         distp = parent_node2(k+1,col);
         sest  = parent_node2(k+1,col+1);
         rm = parent_yp2(:,p);
         rm = rm - sest*R(:,k+1);
         a_est = rm(k,1)/R(k,k);
         sest2 = (abs(a_est - conste).^2);
         distc(1,acu:acu+QRM-1) = distp + sest2;
         vector(acu:acu+QRM-1) = ones(1,QRM)*p;
         pos(acu:acu+QRM-1) = 1:1:QRM;
         parent_yp2(:,p)=rm;
         acu=acu+QRM;
         col=col+2;
       end
       [dist3 ord3]  = sort(distc);
       if(dist3(1)>dmin)
           skip=1;
           break;
       end  
       if(skip==0)
       %  [dist2 ord2]=sort(distc);
         parent_node2t = parent_node2;
         parent_yp2t = parent_yp2;
         col=1;
         for p=1:M,
           parent_node2(k,col:col+1) = [dist3(p) conste(pos(ord3(p)))]; 
           if(vector(ord3(p)) ~= p)
             parent_node2(k+1:nt,col:col+1) = parent_node2t(k+1:nt,(2*vector(ord3(p)))-1:(2*vector(ord3(p))));
             parent_yp2(:,p) = parent_yp2t(:,vector(ord3(p)));
           end  
           col=col+2;
         end
       end  
    end
  end
  if(skip==0)
 % se almacenan los M mejores de la n-esima iteraci?n
    x1(:,indice:indice+(2*M)-1) = parent_node2;  
    indice=indice+(2*M);
    dtotal = x1(1,1:2:indice-1);
    dmin = min(dtotal);
    nodos=nodos+M;
  end
 % se verifica si se abre nuevo el arbol 
   if(n<QRM)
    distp = parent_node(1,row);
    if(distp > dmin)
     break;
    else
     skip=0;
    end 
   end
end
% se determina el vector con menor distancia   
   dtotal = x1(1,1:2:indice-1);
% ordtotal=zeros(1,(indice-1)/2);
contador=1;
for k=2:2:indice-1,
 ordtotal(contador)=k;
 contador=contador+1;
end

   %ordtotal = 2:2:indice-1;
   [dminf ordmin]=sort(dtotal); 
   sest3=x1(:,ordtotal(ordmin(1)));

%    
%    clear dtotal
%    clear ordtotal
 
   
   for k=1:nt
     s_est(index(k),1) = sest3(k);
 end 
end