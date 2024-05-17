function [s_est,nodos] = OSIC_Det(rm,mag2,conste,index)

  nodos=0;  
  [rows cols]=size(mag2);
  dim = length(conste);

%% declaracion
  s_est = zeros(rows,1)+1i*zeros(rows,1);
  sest = zeros(rows,1)+1i*zeros(rows,1);
  sest2 = zeros(1,dim);
%%
  
% se hace la estimaci?n de los s?mbolos   usando el esquema OSIC
  nodos=nodos+1;
  for k = 1:cols
    ind = cols - (k - 1);
    a_est = rm(ind,1)/mag2(ind,ind);
    sest2 = (abs(a_est - conste).^2);
    [dist pos] = sort(sest2);
    sest(ind,1) = conste(pos(1));
    rm = rm - sest(ind,1)*mag2(:,ind);
  end
  
   for k=1:cols
         s_est(index(k),1) = sest(k);
   end 