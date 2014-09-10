function plot_voronoi_probs(P_samples,samples,lam_dim,lam_domain,post_process, nbins)
%
% This makes plots of every pair of marginals (or joint in 2d case) of
% input probability measure defined by P_samples
%
% post_process - is an input that only applies to the 2d case if you want
% to see marginals on a regular grid instead of w.r.t. the Voronoi cells.
%
lam_domain = lam_domain';
if nargin < 5
    post_process = 0;
end
if nargin == 5
    if isempty(post_process)
        post_process = 0;
    end
end
if nargin < 6
    nbins = 20;  % number of bins in each direction
end
if nargin >= 5
    if isempty(nbins)
        nbins = 20;  % number of bins in each direction
    end
end

if lam_dim == 2 % Plot Voronoi tesselations, otherwise plot 2d projections/marginals of the joint inverse measure
    [num_samples,~] = size(samples);
    fig_num = 5; % Starting figure number
    
    fprintf('Plot Voronoi tesselations \n')
    % Add fake samples outside of lam_domain to close Voronoi tesselations at
    % infinity
    midpt = mean(lam_domain);
    for i=1:lam_dim
        new_pt = midpt;
        new_pt(i) = lam_domain(1,i)-10;
        samples = [samples; new_pt];
        new_pt = midpt;
        new_pt(i) = lam_domain(2,i)+10;
        samples = [samples; new_pt];
    end
    
    indices = nchoosek(1:lam_dim,2);
    
    if ~post_process
        % Get convex hulls for Voronoi cells in each plane of input domain
        vv = [];
        cc = [];
        for i=1:nchoosek(lam_dim,2)
            [Vor(i).vertices,Vor(i).cells]=voronoin(samples(:,indices(i,:)));
            %     fprintf('Length of cc = %6i \n', length(cc))
            if length(Vor(i).cells) ~= num_samples+2*lam_dim
                fprintf('plot_voronoi: Houston, we have a problem \n')
                return
            end
        end
        for i=1:nchoosek(lam_dim,2)
            figure(fig_num);
            clf
            fig_num = fig_num+1;            
            hold on
            for l=1:length(Vor(i).cells)
                if all (Vor(i).cells{l}-1 ~= 0)
                    patch(Vor(i).vertices(Vor(i).cells{l},1),Vor(i).vertices(Vor(i).cells{l},2),P_samples(l));
                end
            end
            axis([lam_domain(1,indices(i,1)) lam_domain(2,indices(i,1)) lam_domain(1,indices(i,2)) lam_domain(2,indices(i,2))])
            %title('Voronoi Tesselation colored by probability', 'FontSize', 15)
            s = ['$\lambda_' int2str(indices(i,1)) '$'];
            xlabel(s, 'interpreter', 'latex', 'FontSize', 15)
            s = ['$\lambda_' int2str(indices(i,2)) '$'];
            ylabel(s, 'interpreter', 'latex', 'FontSize', 15)
            colorbar
            hold off
        end
    else
        [num_samples,~] = size(samples);
        % Init the data struct holding info used for marginals
        Lambda_Info = struct('bin_vals',zeros(lam_dim,nbins+1),'bin_ptr',zeros(num_samples,lam_dim));
        for k=1:lam_dim
            % Compute bins in each direction
            Lambda_Info.bin_vals(k,:) = linspace(lam_domain(1,k),lam_domain(2,k),nbins+1);
            % Compute bin pointer for each sample
            [~,Lambda_Info.bin_ptr(:,k)] = histc(samples(:,k),Lambda_Info.bin_vals(k,:));
        end
        indices = nchoosek(1:lam_dim,2);
        Lambda_pairs = struct('pair',struct());
        for i=1:nchoosek(lam_dim,2) % Compute marginals for each 2d pair of coordinates
            Lambda_pairs.pair(i).terms = indices(i,:);
            Lambda_pairs.pair(i).marginal = zeros(nbins+1,nbins+1);
        end
        
        for i = 1:nchoosek(lam_dim,2) % Compute marginals
            for j=1:num_samples
                flag = 0;
                k = 1;
                while k<nbins+1 && flag == 0
                    if Lambda_Info.bin_ptr(j,indices(i,1)) == k
                        kk = 1;
                        while kk<nbins+1 && flag == 0
                            if Lambda_Info.bin_ptr(j,indices(i,2)) == kk
                                Lambda_pairs.pair(i).marginal(kk+1,k+1) = Lambda_pairs.pair(i).marginal(kk+1,k+1) + P_samples(j);
                                flag = 1;
                            end
                            kk = kk+1;
                        end
                    end
                    k = k+1;
                end
            end
        end
        
        fig_num = 10;
        for i = 1:nchoosek(lam_dim,2)
            figure(fig_num);
            zz = [Lambda_pairs.pair(i).marginal(2:end, 2:end) zeros(nbins, 1); zeros(1,nbins+1)];
            surf(Lambda_Info.bin_vals(indices(i,1),:),Lambda_Info.bin_vals(indices(i,2),:),zz)
            shading flat
            axis([lam_domain(1,indices(i,1)) lam_domain(2,indices(i,1)) lam_domain(1,indices(i,2)) lam_domain(2,indices(i,2))])
            s = ['Marginal in $\lambda_' int2str(indices(i,1)) '$ x $\lambda_' int2str(indices(i,2)) '$'];
            %title(s, 'interpreter', 'latex', 'FontSize', 15)
            s = ['$\lambda_' int2str(indices(i,1)) '$'];
            xlabel(s, 'interpreter', 'latex', 'FontSize', 15)
            s = ['$\lambda_' int2str(indices(i,2)) '$'];
            ylabel(s, 'interpreter', 'latex', 'FontSize', 15)
            colorbar
            fig_num = fig_num+1;
        end
    end
    
else % Higher dimension case
    [num_samples,~] = size(samples);
    % Init the data struct holding info used for marginals
    Lambda_Info = struct('bin_vals',zeros(lam_dim,nbins+1),'bin_ptr',zeros(num_samples,lam_dim));
    for k=1:lam_dim
        % Compute bins in each direction
        Lambda_Info.bin_vals(k,:) = linspace(lam_domain(1,k),lam_domain(2,k),nbins+1);
        % Compute bin pointer for each sample
        [~,Lambda_Info.bin_ptr(:,k)] = histc(samples(:,k),Lambda_Info.bin_vals(k,:));
    end
    indices = nchoosek(1:lam_dim,2);
    Lambda_pairs = struct('pair',struct());
    for i=1:nchoosek(lam_dim,2) % Compute marginals for each 2d pair of coordinates
        Lambda_pairs.pair(i).terms = indices(i,:);
        Lambda_pairs.pair(i).marginal = zeros(nbins+1,nbins+1);
        Lambda_pairs.pair(i).box_area = ( Lambda_Info.bin_vals(indices(i,1),2) - Lambda_Info.bin_vals(indices(i,1),1) ) * ( Lambda_Info.bin_vals(indices(i,2),2) - Lambda_Info.bin_vals(indices(i,2),1) );
    end
    
    for i = 1:nchoosek(lam_dim,2) % Compute marginals
        for j=1:num_samples
            flag = 0;
            k = 1;
            while k<nbins+1 && flag == 0
                if Lambda_Info.bin_ptr(j,indices(i,1)) == k
                    kk = 1;
                    while kk<nbins+1 && flag == 0
                        if Lambda_Info.bin_ptr(j,indices(i,2)) == kk
                            Lambda_pairs.pair(i).marginal(kk+1,k+1) = Lambda_pairs.pair(i).marginal(kk+1,k+1) + P_samples(j);
                            flag = 1;
                        end
                        kk = kk+1;
                    end
                end
                k = k+1;
            end
        end
    end
    
    fig_num = 5;
    for i = 1:nchoosek(lam_dim,2)
        figure(fig_num);
        zz = [Lambda_pairs.pair(i).marginal(2:end, 2:end) zeros(nbins, 1); zeros(1,nbins+1)];
        surf(Lambda_Info.bin_vals(indices(i,1),:),Lambda_Info.bin_vals(indices(i,2),:),zz)
        shading flat
        axis([lam_domain(1,indices(i,1)) lam_domain(2,indices(i,1)) lam_domain(1,indices(i,2)) lam_domain(2,indices(i,2))])
        s = ['Marginal in $\lambda_' int2str(indices(i,1)) '$ x $\lambda_' int2str(indices(i,2)) '$'];
        %title(s, 'interpreter', 'latex', 'FontSize', 15)
        s = ['$\lambda_' int2str(indices(i,1)) '$'];
        xlabel(s, 'interpreter', 'latex', 'FontSize', 15)
        s = ['$\lambda_' int2str(indices(i,2)) '$'];
        ylabel(s, 'interpreter', 'latex', 'FontSize', 15)
        colorbar
        fig_num = fig_num+1;
    end
    
end
