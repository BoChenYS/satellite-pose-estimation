%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%           SIMULATED ANNEALING LEAST SQUARES POSE ESTIMATION
%
%
% This package contains the source code which implements the
% Nonlinear Pose Refinement (SA-LMPE) in
%
%       Satellite Pose Estimation with Deep Landmark Regression and 
%                       Nonlinear Pose Refinement  
%                       
%
% The source code, binaries and demo are supplied for academic use only.
% Do not distribute.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [obs, res, inlier_map] = outrem(K, obs, poses, X, th, alpha)
% alpha: remove only predictions with confidence < alpha

m = size(poses,2);
n = size(X,1);

aux = isfinite(cell2mat(obs));
num_obs = sum(aux(1,:));


P = cell(1,m); %projections
for i=1:m
    P{i} = K*[poses(i).R poses(i).t];
end

res = zeros(1,num_obs);
inlier_map = boolean(ones(1,num_obs,'int8'));


k = 1;
for i=1:m
    for j=1:n
        if isnan(obs{i}(1,j))
            continue
        end
        U = [X(j,:)';1];
        u_proj = P{i}*U;
        x = u_proj(1:2)/u_proj(3); % in pixel coordinates
        res(k) = norm(obs{i}(1:2,j) - x); 
        
        if res(k) > th && obs{i}(3,j) < alpha
            obs{i}(1:2,j) = nan;
            inlier_map(k) = 0;
        end
        
        k = k+1;
    end
end
    

end