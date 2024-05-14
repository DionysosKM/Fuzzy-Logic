clear
clc

%split data
data = load('airfoil_self_noise.dat');
[trn_data, val_data, chk_data] = split(data);
performance = zeros(4, 4);

%evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%FIS with grid partition
fis(1) = genfis1(trn_data, 2, 'gbellmf', 'constant'); %singleton
fis(2) = genfis1(trn_data, 3, 'gbellmf', 'constant'); %singleton
fis(3) = genfis1(trn_data, 2, 'gbellmf', 'linear'); %polynomial
fis(4) = genfis1(trn_data, 3, 'gbellmf', 'linear'); %polynomial

%training
for i = 1 : 4
    
    [trnFis, trnError, ~, valFis, valError] = anfis(trn_data, fis(i), [100 0 0.01 0.9 1.1], [], val_data);
    
    %MFs plots 
    for j=1:5 
        figure();
        plotmf(valFis, 'input', j);
        str = ['TSK Model ' , i , ' Feature ' , j];
        title(str);
    end
    
    %learning curve
    figure();
    grid on;
    plot([trnError valError]);
    xlabel('Iterations'); ylabel('Error');
    legend('Training Error','Validation Error');
    str = ['TSK Model ' , i , ' Learning Curve '];
    title(str);
    
    %label prediction
    Y = evalfis(chk_data(:, 1 : end-1), valFis); 
    
    %calculate metrics
    R2 = Rsq(Y, chk_data(:, end)); 
    RMSE = sqrt(mse(Y ,chk_data(:, end)));
    NMSE = 1-R2; 
    NDEI = sqrt(NMSE);
    performance(:, i) = [R2; RMSE; NMSE; NDEI];
    
    %error prediction
    error_prediction = chk_data(:, end) - Y; 
    figure();
    plot(error_prediction);
    grid on;
    xlabel('input'); ylabel('Error');
    str = ['TSK Model ' , i , ' Prediction Error '];
    title(str);
    
end

%results
variables_string = {'TSK_model_1', 'TSK_model_2', 'TSK_model_3', 'TSK_model_4'};
metrics_string = {'Rsquared' , 'RMSE' , 'NMSE' , 'NDEI'};
performance = array2table(performance,'VariableNames',variables_string,'RowNames',metrics_string);