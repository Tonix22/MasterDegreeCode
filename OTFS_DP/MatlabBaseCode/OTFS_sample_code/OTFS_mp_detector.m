%
% Copyright (c) 2018, Raviteja Patchava, Yi Hong, and Emanuele Viterbo, Monash University
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
%ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
%WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
%ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
%SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
%    - Latest version of this code may be downloaded from: https://ecse.monash.edu/staff/eviterbo/
%    - Freely distributed for educational and research purposes
%%

%% chann_taps are the Channel chann_taps
function x_est = OTFS_mp_detector(N,M,M_mod,chann_taps,Delay_taps,Doppler_taps,chan_coef,sigma_2,y)

yv    = reshape(y,N*M,1); % reshape to a vector 
n_ite = 50;
delta = 0.6; % damping factor or convergence speed
alphabet = qammod(0:M_mod-1,M_mod,'gray');% all posible constelations

mean_ec = zeros(N*M,chann_taps);
var_ec  = zeros(N*M,chann_taps);
%M_mod constelation size
%chann_taps, are delay chann_taps

% probability mass function with equally likelihood pero symbol
p_map    = ones(N*M,chann_taps,M_mod)*(1/M_mod); 

conv_rate_prev = -0.1;
z  = exp((1i*2*pi)/(N*M));

for ite=1:n_ite
    %% Update mean and var
    for l=1:1:M % delay
        for k=1:1:N % doppler
            %mean and variance
            % Taps are delay chann_taps s
            mean_de_hat = zeros(chann_taps,1); 
            var_de_hat  = zeros(chann_taps,1);
            % Channel chann_taps
            for e=1:chann_taps
                %Get the channel value given a linear filter iteration Z(q-l)
                %li = Delay_taps(e);
                %ki = Doppler_taps(e);
                % Fixed doppler column k moving along li rows delay path
                % -1 is because taps begin in 0 index
                %l_idx = l-li-1;
                %add_term  = z^(ki*(l_idx));

                %Doppler move left
                %Delay move up
                add_term  = exp(1i*2*pi*((l-Delay_taps(e)-1)/M)*(Doppler_taps(e)/N));
    
                circularity = 1;
                n=0;
                if l-1 < Delay_taps(e) % circularity
                    n = mod(k-Doppler_taps(e)-1,N) + 1;
                    circularity = exp(-1i*2*pi*((n-1)/N));
                end
    
                %dlmwrite('data.csv',[l,k,n,e],'-append') ; % 9 significant figures.

                %Get Channel value H_de
                H_de = add_term * (circularity) * chan_coef(e); 

                %vector position
                %delay is tap and move trought doppler k
                v_pos = N*(l-1)+k;
                
                %Iterate over possible constelations
                for i2=1:1:M_mod
                    mean_de_hat(e) = mean_de_hat(e) + p_map(v_pos,e,i2) * alphabet(i2);
                    var_de_hat(e)  = var_de_hat(e)  + p_map(v_pos,e,i2) * abs(alphabet(i2))^2;
                end
                mean_de_hat(e) = mean_de_hat(e) * H_de;
                var_de_hat(e)  = var_de_hat(e) * abs(H_de)^2;
                var_de_hat(e)  = var_de_hat(e) - abs(mean_de_hat(e))^2;
            end
            % mu_dc_i and sigma_dc_i
            % sum e in I_d
            mean_dc = sum(mean_de_hat);
            var_dc  = sum(var_de_hat)+(sigma_2);
            % End From step 2

            %Matrix each col is a vector of NM where is saved the tap result 
            %There are only tap_no cols or dealy paths
            v_pos = N*(l-1)+k;  
            for e=1:chann_taps
                mean_ec(v_pos,e) = mean_dc - mean_de_hat(e);
                var_ec(v_pos,e)  = var_dc - var_de_hat(e);
            end
        end
    end

    %% Update probabilities
    sum_prob_comp = zeros(N*M,M_mod);
    pmf_delay_shift  = zeros(chann_taps,1);
    pmf_doppler_shift  = zeros(chann_taps,1);

    for l=1:1:M
        for k=1:1:N
            dum_sum_prob = zeros(M_mod,1);
            log_te_var   = zeros(chann_taps,M_mod);
            for tap_no=1:chann_taps
                % Delay Choose
                if l+Delay_taps(tap_no) <= M
                    delay_shift = l+Delay_taps(tap_no);
                    add_term    = exp(1i*2*pi*((l-1)/M)*(Doppler_taps(tap_no)/N));
                    int_flag    = 0;
                else % Delay circularity is needed 
                    delay_shift = l + Delay_taps(tap_no)- M;
                    add_term = exp(1i*2*pi*((l-1-M)/M)*(Doppler_taps(tap_no)/N));
                    int_flag = 1;
                end
                
                % Doppler choose include circularity
                doppler_shift = mod(k-1+Doppler_taps(tap_no),N) + 1;

                k_right_comp = 1;
                % if k_right_comp is needed 
                if int_flag==1
                    k_right_comp = exp(-1i*2*pi*((k-1)/N));
                end

                % Channel H_ec
                H_ec = add_term * k_right_comp * chan_coef(tap_no);

                % PMF taps
                pmf_delay_shift  (tap_no) = delay_shift;
                pmf_doppler_shift(tap_no) = doppler_shift;

                %vector position interleaving because each chann tap shifts change
                v_pos = N*(delay_shift-1)+doppler_shift;
                
                for i2=1:1:M_mod
                    %numerator |-y_e-u_ec-H_ec*a_j|^2
                    dum_sum_prob(i2) = abs(yv(v_pos)- mean_ec(v_pos,tap_no) - H_ec * alphabet(i2))^2;
                    %denominator sigma_ec^2
                    dum_sum_prob(i2) = -(dum_sum_prob(i2)/var_ec(v_pos,tap_no));
                end

                %Reject, exp, sum routine -- Begin 
                %Remove max probability alias exlude e=d
                dum_sum = dum_sum_prob - max(dum_sum_prob);
                %exponential set max elemnet to 1
                dum1 = sum(exp(dum_sum)); % sum xi(e,c,k)
                %Reject, exp, sum routine -- End
                
                log_te_var(tap_no,:) = dum_sum - log(dum1);

            end

            for i2=1:1:M_mod
                ln_qi(i2) = sum(log_te_var(:,i2));
            end

            %Reject, exp, sum routine -- Begin 
            dum_sum = exp(ln_qi - max(ln_qi));
            dum1    = sum(dum_sum);
            %Reject, exp, sum routine -- End

            pcd_aj  = dum_sum/dum1; % 6.27
            sum_prob_comp(N*(l-1)+k,:) = pcd_aj; 

            % PMF update
            for tap_no=1:1:chann_taps
                delay_shift   = pmf_delay_shift(tap_no);
                doppler_shift = pmf_doppler_shift(tap_no);
                v_pos         = N*(delay_shift-1)+doppler_shift;

                dum_sum   = log_te_var(tap_no,:);
                ln_qi_loc = ln_qi - dum_sum;

                %Reject, exp, sum routine -- Begin 
                dum_sum   = exp(ln_qi_loc - max(ln_qi_loc));
                dum1      = sum(dum_sum);
                %Reject, exp, sum routine -- End

                pcd_aj    = dum_sum/dum1; % 6.27
                pcd_aj_before = reshape(p_map(v_pos,tap_no,:),1,M_mod);
                p_map(v_pos,tap_no,:) = (pcd_aj)*delta + (1-delta)*pcd_aj_before; % (37)
            end
        end
    end
    %Step 3 Convergence indicator
    conv_rate =  sum(max(sum_prob_comp,[],2) > 0.99)/(N*M); % (6.29)(39)
    %Step 4 
    if conv_rate==1
        sum_prob_fin = sum_prob_comp;
        break;
    elseif conv_rate > conv_rate_prev
        conv_rate_prev = conv_rate;
        sum_prob_fin = sum_prob_comp;
    elseif (conv_rate < conv_rate_prev - 0.2) && conv_rate_prev > 0.95
        break;
    end
end
x_est = zeros(N,M);
for l=1:1:M
    for k=1:1:N
        [~,pos] = max(sum_prob_fin(N*(l-1)+k,:));%argmax 
        x_est(k,l) = alphabet(pos); % guessing the sybol
    end
end
end