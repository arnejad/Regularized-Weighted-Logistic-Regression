% This code has been developed by Ali Rahmani Nejad at IASBS for educational puporses

%% Initialization
% feats = demo2x{:,:};
% lbls = demo2y{:,:};

feats = dlmread('features.txt');
fid = fopen('labels.txt');
lbls = fscanf(fid, '%f');  % column one coresponds to ...
                                % X (feature) and 2nd column Y (lables)
test_sample = [-0.6 0.6];

test_sample_phi = phiMaker(test_sample(1), test_sample(2));

% X = [ones(size(feats, 1),1),feats];
X = phiMaker(feats(:,1), feats(:,2));
m = length(lbls);

theta = double(zeros(size(X,2), 1));
pos = find(lbls == 1);
neg = find(lbls == 0);

lambda = 0.0001;

I = eye(length(theta));
I(1,1) = 0;

PARAM = 1;
W = diag(exp(-(((dist(test_sample_phi, X').^2)/(2 * (PARAM)^2)))));

%% Newton-Rhapson Method
for i=0:5
    sig = sigmoid(X * theta);
    J = sum(-lbls .* log(sig) - (1-lbls) .* log(1-sig))/m + (lambda * (theta' * theta))/(2*m);
    Hess = (X' * (W * diag(sig .* (1-sig))) * X)/m + (lambda * I)/m;
    dJ = sum((sig - lbls)' .* X', 2)/m - lambda * theta/m;
    grad = pinv(Hess) * dJ;    
    theta = theta - grad;
end

u = linspace(-1, 1, 200);
v = linspace(-1, 1, 200);
[uu , vv] = meshgrid(u, v);

plotz = zeros(length(uu));

for i=1:length(uu)
   z = theta' * phiMaker(uu(i,:)', vv(i,:)')';
   for j=1:length(z)
      plotz(i,j) = z(1,j); 
   end
end


%% Visualization
figure(1);
 plotz(plotz>-1.368788969284922e+18) = 100;
plotz(plotz<-1.368788969284922e+18) = -100;
contourf(uu,vv,plotz)
hold on;
RGB = [204 204 255]/256 ;

scatter(feats(pos,1), feats(pos,2), 50, 'MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .0 .7])

scatter(feats(neg,1), feats(neg,2), 50, 'MarkerEdgeColor',[0 0 1],...
              'MarkerFaceColor',[1 0 0])
          
scatter(test_sample(1), test_sample(2), 50, 'MarkerEdgeColor',[0 0 0],...
              'MarkerFaceColor',[0 0 0])
title("simple linear regression (PARAM: " + PARAM + ")")
