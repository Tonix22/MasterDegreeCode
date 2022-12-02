
NORMALIZE = false;
UNITPOWER = false;

file = fopen('QAM_arrays.py', 'w+');
fprintf(file,'import numpy as np\r\n');
fprintf(file,'\r\n');

for u = ["Data","Unit_Pow","Norm"]
    for v = [4,16, 32, 64]
        x = (0:v-1)';
        if u == 'Unit_Pow'
            y = qammod(x,v,'UnitAveragePower',true);
        elseif u == 'Norm'
            y = qammod(x,v);
            y = real(y)/max(real(y))+j*imag(y)/max(imag(y));
        else
            y = qammod(x,v);
        end
        
        fprintf(file,'QAM_%d_%s = np.array([',v,u);
    
        for i = 1:v
            fprintf(file,'%d+%dj,',real(y(i)),imag(y(i)));
        end
        fprintf(file,'])\r\n');
    end
    fprintf(file,'\r\n');
    fprintf(file,'##########\r\n');
    fprintf(file,'\r\n');
    
end

fclose(file);

