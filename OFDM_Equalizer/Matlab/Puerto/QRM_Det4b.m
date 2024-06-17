function s_est  = QRM_Det4b(yp,R,conste,index)
  [row nt] = size(R);
  QRM = length(conste);
  opercomp=0;
  dim=length(conste);
  
  %%  dimensiones
  s_est = complex(zeros(nt,1));

% sest = complex(zeros(1,1));
  sest2 = zeros(1,QRM);
  sest3 = complex(zeros(nt,1));
  sestm = complex(zeros(nt,QRM));
 

  
  
  %%
  nodos=0;
% se realiza la detecci?n del nivel nt  
  a_est = yp(nt,1)/R(nt,nt);  
  sest2 = (abs(a_est - conste).^2);
  [dist ord]=sort(sest2);
  opercomp = opercomp+dim+(dim*3)+1;

  dmin=inf;
  skip=0;
  idx=1;
  distm = zeros(1,QRM);
  distm2 = zeros(1,QRM);
for n=1:QRM,     
  branches=QRM;
  contador2=1;
  
  
%   clear parent_dist; %(1,QRM)-->0
%   clear parent_node;%(1,QRM)-->0
%   clear parent_yp;%(48,QRM)-->0

  parent_dist = zeros(nt,QRM);
  parent_node = complex(zeros(48,QRM));
  parent_yp = complex(zeros(48,QRM));
  
  
  
  
  %distm = inf;
%   distm2 = inf;

  % se realiza la estimaci?n de los nt-1 niveles
  for k=nt-1:-1:1,
    if(k==nt-1)
       distp = dist(n);
       sest = conste(ord(n));
       opercomp = opercomp+2;
       rm = yp;
       rm = rm - sest*R(:,k+1);
       a_est = rm(k,1)/R(k,k);
       sest2 = (abs(a_est - conste).^2);
       distc2 = distp + sest2;  
       [dist2 ord2]=sort(distc2);
       opercomp = opercomp+dim+(dim*3)+1;
       opercomp = opercomp+2*(k-1);
       for p=1:QRM,
        parent_dist(k+1,p) = dist(n);
        parent_node(k+1,p) = conste(ord(n));
        parent_dist(k,p) = dist2(p);
        parent_node(k,p) = conste(ord2(p));
        parent_yp(:,p) = rm;
        opercomp = opercomp+2;
       end
    else
%        clear distc; %(1,QRM) -->0
%        clear child_dist; % (1,QRM^2-->inf)
%        clear child_pos; % (1,QRM^2)-->0
%        clear child_node; % (1,QRM^2)-->0
%   
       
 distc = zeros(1,QRM);
 child_dist = zeros(1,QRM^2);
 %distm = ones(1,QRM^2)*inf;
% distm2 = ones(1,QRM^2)*inf;


 child_pos = zeros(1,QRM^2);
 child_node = complex(zeros(1,QRM^2)); 
 vector = zeros(1,QRM^2);      
       
       contador=1;
       acu=1;
       for p=1:QRM,
         distp = parent_dist(k+1,p);
         sest  = parent_node(k+1,p);
         opercomp = opercomp+2;
         rm = parent_yp(:,p);
         rm = rm - sest*R(:,k+1);
         a_est = rm(k,1)/R(k,k);
         sest2 = (abs(a_est - conste).^2);
         distc = distp + sest2;
         [dist3 ord3] = sort(distc);
         child_dist(acu:acu+QRM-1) = dist3;
         child_pos(acu:acu+QRM-1) = ord3;
         child_node(acu:acu+QRM-1) = conste(ord3);
         parent_yp(:,p)=rm;
         vector(acu:acu+QRM-1) = ones(1,QRM)*p;
         acu=acu+QRM;
         opercomp = opercomp+dim+(dim*3)+1;
         opercomp = opercomp+2*(k-1);
       if(p<QRM)
           if(k>16)  
             if(parent_dist(k+1,p+1) > min(child_dist(1:QRM*p)))
%            if(parent_dist(k+1,p+1) > min(child_dist))
              contador=contador+1;
              if(contador==3)
               break;
              end
             end
           else
             if(parent_dist(k+1,p+1) > min(child_dist(1:QRM*p)))
              break;
             end
           end
         end
       end
       [dist4 ord4]  = sort(child_dist(1:QRM*p));
       %  [dist4 ord4]  = sort(child_dist);
       opercomp = opercomp+acu;
       if(dist4(1)>dmin)
           skip=1;
           break;
       else
         parent_nodet = parent_node;
         parent_distt = parent_dist;
         parent_ypt = parent_yp;
         for p=1:QRM,
           parent_dist(k,p) = dist4(p);
           parent_node(k,p) = conste(child_pos(ord4(p)));
           if(vector(ord4(p)) ~= p)
             parent_node(k+1:nt,p) = parent_nodet(k+1:nt,vector(ord4(p)));
             parent_dist(k+1:nt,p) = parent_distt(k+1:nt,vector(ord4(p)));
             parent_yp(:,p) =  parent_ypt(:,vector(ord4(p)));
             opercomp = opercomp+2;
           end
         end
       end
    end
  end
  if(skip==0)
 % se almacenan el mejor vector de los M construidos en la n-esima iteraci?n
    dtotal = parent_dist(:,1:1:QRM);
    [dmin dpos] = min(dtotal(1,:));
    sestm(:,idx) = parent_node(:,dpos); 
    distm(1,idx) = parent_dist(1,dpos);
    distm2(1,idx) = parent_dist(32,dpos);
    idx=idx+1;
    nodos=nodos+branches;  
  end
 % se verifica si se abre nuevo el arbol 
   if(n<QRM)
    %if(dist(n+1) > min(distm))
    if(dist(n+1) > min(distm2(1:idx-1))) 
   %   contador2=contador2+1;   
    %  if(contador2==2)
        break;
    %  end
    else
     skip=0;
    end 
   end
end

% se determina el vector con menor distancia   
   
   [dminf ordmin]=sort(distm(1:idx-1)); 
   sest3=sestm(:,ordmin(1));
 
   for k=1:nt
     s_est(index(k),1) = sest3(k);
 end 

end