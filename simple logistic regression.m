clear all
%% INITIALIZATIONS
% fid = fopen('features.txt');
feats = dlmread('features.txt');
fid = fopen('labels.txt');
labels = fscanf(fid, '%f');  % column one coresponds to ...
                                % X (feature) and 2nd column Y (lables)

feats_train = feats(1:60,:);
feats_test = feats(61:69,:);

lbls_train = labels(1:60,:);
lbls_test = labels(61:69,:);

clear fid feats labels;

lr=0.1;
num_iter=10000;

%% TRAIN
theta = zeros(1,2);

% gradiant descend
for i=1:num_iter
%     z = np.dot(X, theta)
%     h = sigmoid(z)
%     gradient = -np.dot(X.T, (h - y)) / y.size
%     theta = theta+ lr * gradient

    z = theta*feats_train';
    h = sigmoid(z);
    gradient = - ((h - lbls_train') * feats_train) / length(lbls_train);
    theta = theta + lr * gradient;
end

figure(1);
scatter(feats_train(:,1), feats_train(:,2), 128,lbls_train,'filled');
hold on;
fplot(poly2sym(theta), 'red') % draw the function