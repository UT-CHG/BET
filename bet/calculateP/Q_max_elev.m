add_obs_matrix = addElevStation(Domain,[1900, 300; 1900, 1200; 1900, -300; 1900, 600]);

true_param_val = 10401; %defines true parameter value used to define noisy data

N = num_fields; %defines number N samples in Lambda used to define rho_{Lambda,N,M}

Q = zeros(N,4);

global Q Q_num
Q_num = [1 3];
for i=1:N
    Q(i,:) = ( add_obs_matrix*maxele63(:,i) )';
end

lam_domain = [0.07, 0.15; .03, 0.08; 0.1, 0.2];
lam_dim = 3;
lam_samples = mann_pts(1:3,1:N)';

M = 3E3; %defines number M samples in D used to define rho_{D,M}

clear q_distr_samples
% Create M samples defining M bins in D used to define rho_{D,M}
for i=1:numel(Q_num)
    if Q_num(i) < 3
        q_distr_samples(:,i) = (Q(true_param_val,Q_num(i)))-0.05+0.1*rand(M,1);
    else
        q_distr_samples(:,i) = (Q(true_param_val,Q_num(i)))-0.01+0.02*rand(M,1);
    end
end
% Now compute probabilities for rho_{D,M} by sampling from rho_D
% First generate samples of rho_D
num_q_emulate = 1E6;
clear q_distr_emulate
for i=1:numel(Q_num)
    if Q_num(i) < 3
        q_distr_emulate(:,i) = (Q(true_param_val,Q_num(i)))-0.035+0.07*rand(num_q_emulate,1);
    else
        q_distr_emulate(:,i) = (Q(true_param_val,Q_num(i)))-0.0075+0.015*rand(num_q_emulate,1);
    end
end
% Now bin samples of rho_D in the M bins of D to compute rho_{D,M}
tic
dist_vector = zeros(M,1);
count_neighbors = zeros(M,1);
for i=1:num_q_emulate
    dist_vector(:) = sum( bsxfun(@minus,q_distr_emulate(i,:)',q_distr_samples').^2, 1 );
    [~,Itemp] = min(dist_vector);
    count_neighbors(Itemp(1)) = count_neighbors(Itemp(1))+1;
end
toc
q_distr_prob = count_neighbors / (num_q_emulate); %This together with q_distr_samples defines rho_{D,M}

% Determine which inputs go to which M bins using the QoI
dist_vector = zeros(M,1);
io_ptr = zeros(N,1);
% count_neighbors = zeros(M,1);
for i=1:N
    dist_vector(:) = sum( bsxfun(@minus,Q(i,Q_num)',q_distr_samples').^2, 1 );
    [~,Itemp] = min(dist_vector);
    io_ptr(i) = Itemp(1);
%     count_neighbors(Itemp(1)) = count_neighbors(Itemp(1))+1;
end

% [io_ptr,sorted_ind] = sort(io_ptr);
P = zeros(N,1);
lam_vol = ones(N,1);
tic
for i=1:M
    Itemp = find(io_ptr == i);
    if ~isempty(Itemp)
        P(Itemp) = lam_vol(Itemp)/(sum(lam_vol(Itemp))) * q_distr_prob(i);
    end
end
toc
P = P/sum(P);

% [~,~,P] = BET(@Q_choice,N,lam_dim,lam_domain,20,lam_samples,[],lam_samples,Q(:,Q_num),q_distr_samples,q_distr_prob);

plot_voronoi_probs(P,lam_samples,lam_dim,lam_domain);



figure;
ind = find(P > 1E-4);
scatter3(lam_samples(ind,1),lam_samples(ind,2),lam_samples(ind,3),100,P(ind),'*')

