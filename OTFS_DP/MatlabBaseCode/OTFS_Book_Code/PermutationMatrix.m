N = 2;
M = 3;
P=zeros(N*M,N*M);
for j=1:N
    for i=1:M
        E=zeros(M,N);
        E(i,j)=1;
        P((j-1)*M+1:j*M,(i-1)*N+1:i*N) = E;
    end
end