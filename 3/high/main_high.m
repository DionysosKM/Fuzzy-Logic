clear
clc

%load and normalize data
data = csvread('superconduct.csv', 1, 0);
normalized_data = data(:, 1 : end-1);
normalized_data = normalize(normalized_data);
data = [normalized_data(:, 1 : end) data(:, end)];

%evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%best model
features = 12;
ra = 0.4;

%most useful features 
[index, weights] = relieff(data(:, 1 : end-1),data(:, end), 6); 

%save metrics
metrics = zeros(5, 4); 

%split data - 80% train, 20% test       
train_test = cvpartition(data(:, end), 'KFold', 5, 'Stratify', true); 

first_time = true; 
plot_features_indexes = [1, 3, 8, 12];

for i = 1 : train_test.NumTestSets
            
    %load train, test data 
    data_80 = data(training(train_test, i), :);
    chk_data = data(test(train_test, i), :);

    %75%  of training data for training, 25% for evaluation
    train_val = cvpartition(data_80(:, end), 'KFold', 4, 'Stratify', true);
    trn_data = data_80(training(train_val, 2), :);
    val_data = data_80(test(train_val, 2), :);

    %most useful features
    trn_data = [trn_data(:, index(1 : features)) trn_data(:, end)];
    val_data = [val_data(:, index(1 : features)) val_data(:, end)];
    chk_data = [chk_data(:, index(1 : features)) chk_data(:, end)];
    
    %FIS        
    fis = genfis2(trn_data(:,1:end-1), trn_data(:,end), ra);
    
    %plot MFs before training, once 
    if first_time == true
        
        for j = 1 : length(plot_features_indexes)
            figure();
            plotmf(fis, 'input', plot_features_indexes(j));
            s_1 = "Feature  ";
            s_2 = num2str(plot_features_indexes(j));
            s_3 = " MF before training";
            title_1 = strcat(s_1, s_2);
            title_2 = strcat(title_1, s_3);
            title(title_2);
        end
        
        first_time = false;
        
    end
    
    %train
    [trn_fis, trn_error, ~, val_fis, val_error] = anfis(trn_data, fis, [100 0 0.01 0.9 1.1], [], val_data); 
    
    %metrics
    Y = evalfis(chk_data(:, 1 : end-1), val_fis);
    RMSE = sqrt(mse(Y, chk_data(:, end)));
    R2 = Rsq(Y, chk_data(:, end));
    NMSE = 1 - R2; 
    NDEI = sqrt(NMSE);
          
    metrics(i, 1) = RMSE;
    metrics(i, 2) = NMSE;
    metrics(i, 3) = NDEI;
    metrics(i, 4) = R2;
    
end

%mean average of each metric      
RMSE = sum(metrics(:, 1))/5; 
NMSE = sum(metrics(:, 2))/5; 
NDEI = sum(metrics(:, 3))/5; 
R2 = sum(metrics(:, 4))/5; 

%array for plots
y = zeros(size(Y, 1), 1);
for i= 1:size(Y, 1)
    y(i) = i;
end

%plot MFs after training
for j = 1 : length(plot_features_indexes)
    figure();
    plotmf(val_fis, 'input', plot_features_indexes(j));
    s_1 = "Feature  ";
    s_2 = num2str(plot_features_indexes(j));
    s_3 = " MF after training";
    title_1 = strcat(s_1, s_2);
    title_2 = strcat(title_1, s_3);
    title(title_2);
end

%predicted values
figure();
scatter(y, Y); grid on;
xlabel('data'); ylabel('predicted values');
title('predicted values');

%real values
figure();
scatter(y, chk_data(:, end)); grid on;
xlabel('data'); ylabel('Real Values');
title('Real values');

%learning curve and errors
figure();
grid on;
plot([trn_error val_error]);
xlabel('iterations');
ylabel('error');
legend('training error', 'validation error');
t = "Learning curve";
title(t);