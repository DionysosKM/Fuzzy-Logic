close all;
clear all;

%load and normalize data
data = csvread('superconduct.csv');
normalized_data = data(:, 1 : end-1);
normalized_data = normalize(normalized_data);
data = [normalized_data(:, 1 : end) data(:, end)];

%evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%grid 
num_features = 4;          
num_ra = 7;

grid_parameters = zeros(num_features, num_ra, 2);

grid_parameters(1,:,1) = 3;
grid_parameters(2,:,1) = 6;
grid_parameters(3,:,1) = 9;
grid_parameters(4,:,1) = 12;
grid_parameters(:,1,2) = 0.2;
grid_parameters(:,2,2) = 0.3;
grid_parameters(:,3,2) = 0.4;
grid_parameters(:,4,2) = 0.5;
grid_parameters(:,5,2) = 0.6;
grid_parameters(:,6,2) = 0.7;
grid_parameters(:,7,2) = 0.8;

%save metrics 
metrics = zeros(num_features, num_ra, 4);  

%most useful features 
[index, weights] = relieff(data(:, 1 : end-1),data(:, end), 6); % ranking of features 

%arrays for RMSE, number of rules, and selected number of features
rmse = zeros(num_features, num_ra);
num_rules = zeros(num_features, num_ra);
selected_features = zeros(num_features, num_ra);

%grid search
for i = 1 : num_features    
    for j = 1 : num_ra 
        
        features = grid_parameters(i, j, 1);
        ra = grid_parameters(i, j, 2);
       
        %split data - 80% train, 20% test        
        partition_train_test = cvpartition(data(:, end), 'KFold', 5, 'Stratify', true);    
        
        %save metrics (4) of cross validation, 5 because 5-fold
        model_metrics = zeros(5, 4);  
        
        for k = 1 : partition_train_test.NumTestSets
            
           %load train, test data 
           data_80 = data(training(partition_train_test, k),:);
           chk_data = data(test(partition_train_test, k), :);
           
           %75% of training data for training, 25% for evaluation           
           partition_train_val = cvpartition(data_80(:, end), 'KFold', 4, 'Stratify', true);
           trn_data = data_80(training(partition_train_val, 2), :);
           val_data = data_80(test(partition_train_val, 2), :);
           
           %save the most useful features  
           trn_data = [trn_data(:, index(1 : features)) trn_data(:, end)];
           val_data = [val_data(:, index(1 : features)) val_data(:, end)];
           chk_data = [chk_data(:, index(1 : features)) chk_data(:, end)];
           
           %FIS with subtractive clustering     
           fis = genfis2(trn_data(:, 1 : end-1), trn_data(:, end), ra);
           
           [trn_fis, trn_error, ~, val_fis, val_error] = anfis(trn_data, fis, [100 0 0.01 0.9 1.1], [], val_data); 
           
           %calculate and save metrics
           Y = evalfis(chk_data(:, 1 : end-1), val_fis);
           RMSE = sqrt(mse(Y, chk_data(:, end)));
           R2 = Rsq(Y, chk_data(:, end));
           NMSE = 1 - R2; 
           NDEI = sqrt(NMSE);
           
           model_metrics(k, 1) = RMSE;
           model_metrics(k, 2) = NMSE;
           model_metrics(k, 3) = NDEI;
           model_metrics(k, 4) = R2;
           
        end
        
        %mean average of each metric        
        metrics(i, j, 1) = sum(model_metrics(:, 1))/5; % RMSE
        metrics(i, j, 2) = sum(model_metrics(:, 2))/5; % NMSE
        metrics(i, j, 3) = sum(model_metrics(:, 3))/5; % NDEI
        metrics(i, j, 4) = sum(model_metrics(:, 4))/5; % R2
        
        %save RMSE, number of rules, and selected number of features        
        rmse(i, j) = metrics(i, j, 1);
        num_rules(i, j) = size(val_fis.rule, 2);
        selected_features(i, j) = features;

    end
end

%number of rules against RMSE
figure();
scatter(reshape(rmse, 1, []),reshape(num_rules, 1, [])); grid on;
xlabel("RMSE"); 
ylabel("Number of rules");
title("RMSE against number of rules ");

%number of features against RMSE
figure();
scatter(reshape(rmse, 1, []),reshape(selected_features, 1, [])); grid on;
xlabel("RMSE"); 
ylabel("Number of features");
title("RMSE against number of features");