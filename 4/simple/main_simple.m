clc
clear

%load and split data
data = load('haberman.data');
[trn_data, val_data, chk_data] = split(data);

%extreme values for radius to get different number of rules
ra = [0.1 0.9];

%initializing perfromance tables
OA_dep = zeros(2, 1);
PA_dep = zeros(2, 2);
UA_dep = zeros(2, 2);
K_dep = zeros(2, 1);
error_matrix_dep = zeros(2, 2, 2);
rules_number_dep = zeros(2, 1);

OA_indep = zeros(2, 1);
PA_indep = zeros(2, 2);
UA_indep = zeros(2, 2);
K_indep = zeros(2, 1);
error_matrix_indep = zeros(2, 2, 2);
rules_number_indep = zeros(2, 1);

%4 models, one class dependent and one class independent for each radius'
%value
for i = 1:2

    %clustering per class
    [c_1, sig_1] = subclust(trn_data(trn_data(:, end)==1, :), ra(i));
    [c_2, sig_2] = subclust(trn_data(trn_data(:, end)==2, :), ra(i));
    num_rules = size(c_1, 1)+size(c_2, 1);
    
    %FIS   
    fis = newfis('FIS_SC','sugeno');
    
    %input variables 
    names_in = {'in1', 'in2', 'in3'};   
    for j = 1 : size(trn_data, 2)-1           
        fis = addvar(fis, 'input', names_in{j}, [0 1]);         
    end
    
    %output variables
    fis = addvar(fis, 'output', 'out1', [0 1]);
    
    %input Membership Functions
    name = 'MF';
    
    for j = 1 : size(trn_data, 2)-1      
        count = 1;    
        
        for m = 1 : size(c_1, 1)            
            fis = addmf(fis, 'input', j, "MF " + count, 'gaussmf', [sig_1(j) c_1(m, j)]);
            count = count+1;           
        end
        
        for m = 1 : size(c_2, 1)         
            fis = addmf(fis, 'input', j, "MF " + count, 'gaussmf', [sig_2(j) c_2(m, j)]);
            count = count+1;           
        end        
    end
    
    %output Membership Functions
    parameters = [zeros(1, size(c_1, 1)) ones(1, size(c_2, 1))];
    
    for j = 1 : num_rules  
        fis = addMF(fis, 'out1', 'constant', parameters(j));     
    end
    
    %FIS rule base  
    rules = zeros(num_rules, size(trn_data, 2));
    
    for j = 1 : size(rules, 1)       
        rules(j, :) = j;      
    end
    
    rules = [rules ones(num_rules, 2)];
    fis = addrule(fis, rules);
    
    %train and evaluate ANFIS 
    [~, trn_error, ~, val_fis, val_error] = anfis(trn_data, fis, [100 0 0.01 0.9 1.1], [], val_data);
    figure();
    plot([trn_error val_error],'LineWidth',2); grid on;
    legend('Training Error','Validation Error');
    xlabel('Epochs');
    ylabel('Error');
    Y=evalfis(chk_data(:, 1 : end-1), val_fis);
    Y=round(Y);
    
    %make Y discrete
    for j = 1 : size(Y, 1)        
        if Y(j) < 1      
            Y(j) = 1;           
        elseif Y(j) > 2          
            Y(j) = 2;          
        end        
    end

    %plot class dependent models' MFs 
    for j = 1 : size(trn_data,2)-1        
        figure();
        plotmf(val_fis, 'input', j);
        t = "Class dependent model, radius = " + ra(i) + " , Feature " + j;
        title(t);        
    end
    
    %error matrix 
    error_matrix = confusionmat(chk_data(:, end), Y);
    
    %overall accuracy 
    N = size(chk_data, 1);
    OA = (error_matrix(1, 1) + error_matrix(2, 2))/N;
    
    %producer's accuracy for each class 
    PA_1 = error_matrix(1, 1)/(error_matrix(1, 1)+error_matrix(1, 2));
    PA_2 = error_matrix(2, 2)/(error_matrix(2, 2)+error_matrix(2, 1));
    
    %user's accuracy for each class
    UA_1 = error_matrix(1, 1)/(error_matrix(1, 1)+error_matrix(2, 1));
    UA_2 = error_matrix(2, 2)/(error_matrix(2, 2)+error_matrix(1, 2));
    
    %K   
    K = ( N*(error_matrix(1, 1)+error_matrix(2, 2))-((error_matrix(1, 1)+error_matrix(2, 1))*(error_matrix(1, 1)+error_matrix(1, 2))+(error_matrix(1, 2)+error_matrix(2, 2))*(error_matrix(2, 1)+error_matrix(2, 2))))/(N^2-((error_matrix(1, 1)+error_matrix(2, 1))*(error_matrix(1, 1)+error_matrix(1, 2))+(error_matrix(1, 2)+error_matrix(2, 2))*(error_matrix(2, 1)+error_matrix(2, 2))));  
    
    %class dependent models' metrics  
    OA_dep(i, 1) = OA;
    PA_dep(i, 1) = PA_1;
    PA_dep(i, 2) = PA_2;
    UA_dep(i, 1) = UA_1;
    UA_dep(i, 2) = UA_2;
    K_dep(i, 1) = K;
    error_matrix_dep(:, :, i) = error_matrix;
    
    %number of rules 
    rules_number_dep(i, 1) = size(val_fis.rule, 2);
    
    %Compare with Class-Independent Scatter Partition 
    fis2 = genfis2(trn_data(:, 1 : end-1), trn_data(:, end), ra(i));
    [trnFis, trn_error, ~, val_fis, val_error] = anfis(trn_data, fis2, [100 0 0.01 0.9 1.1], [], val_data);
    figure();
    plot([trn_error val_error], 'LineWidth', 2); grid on;
    legend('Training Error', 'Validation Error');
    xlabel('# of Epochs');
    ylabel('Error');
    Y = evalfis(chk_data(:, 1 : end-1), val_fis);
    Y = round(Y);
    
    %need to transform array Y so the only values are 1 and 2 (same as
    %classes)
    for j = 1 : size(Y, 1)       
        if Y(j) < 1           
            Y(j) = 1;            
        elseif Y(j) > 2            
            Y(j) = 2;           
        end        
    end
    
    %plot class independent models' MFs for each feature 
    for j = 1:size(trn_data, 2)-1       
        figure();
        plotmf(val_fis, 'input', j);
        t = "Class independent model, radius = " + ra(i) + " , Feature " + j;
        title(t);       
    end
    
    %error matrix 
    error_matrix = confusionmat(chk_data(:, end), Y);
    
    %overall accuracy 
    N = size(chk_data, 1);
    OA = (error_matrix(1, 1) + error_matrix(2, 2))/N;
    
    %producer's accuracy for each class 
    PA_1 = error_matrix(1, 1)/(error_matrix(1, 1) + error_matrix(1, 2));
    PA_2 = error_matrix(2, 2)/(error_matrix(2, 2) + error_matrix(2, 1));
    
    %user's accuracy for each class  
    UA_1 = error_matrix(1, 1)/(error_matrix(1, 1) + error_matrix(2, 1));
    UA_2 = error_matrix(2, 2)/(error_matrix(2, 2) + error_matrix(1, 2));
    
    %K   
    K = ( N*(error_matrix(1, 1)+error_matrix(2, 2))-((error_matrix(1, 1)+error_matrix(2, 1))*(error_matrix(1, 1) + error_matrix(1, 2))+(error_matrix(1, 2)+error_matrix(2, 2))*(error_matrix(2, 1)+error_matrix(2, 2))))/(N^2-((error_matrix(1, 1)+error_matrix(2, 1))*(error_matrix(1, 1)+error_matrix(1, 2))+(error_matrix(1, 2)+error_matrix(2, 2))*(error_matrix(2, 1)+error_matrix(2, 2))));  
    
    %independent models' metrics   
    OA_indep(i, 1) = OA;
    PA_indep(i, 1) = PA_1;
    PA_indep(i, 2) = PA_2;
    UA_indep(i, 1) = UA_1;
    UA_indep(i, 2) = UA_2;
    K_indep(i, 1) = K;
    error_matrix_indep(:, :, i) = error_matrix;
    
    %number of rules  
    rules_number_indep(i, 1) = size(val_fis.rule, 2);
    
end  