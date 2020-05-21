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

function [ res] = reproj_error(K, obs, poses, X)

m = size(poses,2);
n = size(X,1);

aux = isfinite(cell2mat(obs));
num_obs = sum(aux(1,:));


P = cell(1,m); %projections
for i=1:m
    P{i} = K*[poses(i).R poses(i).t];
end

res = zeros(1,num_obs);

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
        
        
        k = k+1;
    end
end
    

end