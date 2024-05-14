close all;
clear all;

%load and normalize data
data = csvread('epileptic_seizure_data.csv', 1, 1);
norm_data = data(:, 1 : end-1);
norm_data = normalize(norm_data);
data = [norm_data(:, 1 : end) data(:, end)];

%evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%best model 
features = 12;
ra = 0.8;

%most useful features 
[index, weights] = relieff(data(:, 1 : end-1),data(:, end), 6); 

first_time = true; 
plot_features_indexes = [1, 3, 8, 12];

%initialize arrays for speed
PA_1_arr(5) = zeros;
PA_2_arr(5) = zeros;
PA_3_arr(5) = zeros;
PA_4_arr(5) = zeros;
PA_5_arr(5) = zeros;
UA_1_arr(5) = zeros;
UA_2_arr(5) = zeros;
UA_3_arr(5) = zeros;
UA_4_arr(5) = zeros;
UA_5_arr(5) = zeros;
OA_arr(5) = zeros;
K_arr(5) = zeros;

%split data - 80% train, 20% test        
train_test = cvpartition(data(:, end), 'KFold', 5, 'Stratify', true); 

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
           
           %clustering per class          
           [c_1, sig_1] = subclust(trn_data(trn_data(:, end)==1, :), ra);
           [c_2, sig_2] = subclust(trn_data(trn_data(:, end)==2, :), ra);
           [c_3, sig_3] = subclust(trn_data(trn_data(:, end)==3, :), ra);
           [c_4, sig_4] = subclust(trn_data(trn_data(:, end)==4, :), ra);
           [c_5, sig_5] = subclust(trn_data(trn_data(:, end)==5, :), ra);
           num_rules = size(c_1, 1)+size(c_2, 1)+size(c_3, 1)+size(c_4, 1)+size(c_5, 1);
           
           %FIS          
           fis = newfis('FIS_SC','sugeno');
           
           %add Input-Output Variables           
           for m = 1 : size(trn_data, 2)-1               
               fis = addvar(fis, 'input', "in " +  m,[0 1]);               
           end
           
           fis = addvar(fis, 'output', 'out1', [0 1]);
           
           %add Input Membership Functions
           for m = 1 : size(trn_data, 2)-1

               count = 1;

               for n = 1 : size(c_1, 1)
                   fis = addmf(fis, 'input', m, "MF " + count, 'gaussmf', [sig_1(m) c_1(n,m)]);
                   count = count+1;
               end

               for n = 1 : size(c_2, 1)
                   fis = addmf(fis, 'input', m, "MF " + count, 'gaussmf', [sig_2(m) c_2(n,m)]);
                   count = count+1;
               end
               
               for n = 1 : size(c_3, 1)
                   fis = addmf(fis, 'input', m, "MF " + count, 'gaussmf', [sig_3(m) c_3(n,m)]);
                   count = count+1;
               end
               
               for n = 1 : size(c_4, 1)
                   fis = addmf(fis, 'input', m, "MF " + count, 'gaussmf', [sig_4(m) c_4(n,m)]);
                   count = count+1;
               end
               
               for n = 1 : size(c_5, 1)
                   fis = addmf(fis, 'input', m, "MF " + count, 'gaussmf', [sig_5(m) c_5(n,m)]);
                   count = count+1;
               end

           end
           
           %add Output Membership Functions          
           params = [zeros(1, size(c_1, 1)) 0.25*ones(1, size(c_2, 1)) 0.5*ones(1, size(c_3, 1)) 0.75*ones(1, size(c_4, 1)) ones(1, size(c_5, 1))];
           
           for m = 1 : num_rules             
               fis = addmf(fis, 'output', 1, "MF " + m, 'constant', params(m));               
           end
           
           %add FIS Rule Base          
           rules = zeros(num_rules, size(trn_data, 2));
           
           for m = 1 : size(rules, 1)              
               rules(m, :) = m;               
           end
           
           rules = [rules ones(num_rules, 2)];           
           fis = addrule(fis, rules);
           
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
           
           %train and evaluate ANFIS
           [trn_fis, trn_error, ~, val_fis, val_error] = anfis(trn_data, fis, [100 0 0.01 0.9 1.1],[], val_data);
           Y = evalfis(chk_data(:, 1 : end-1), val_fis);
           Y = round(Y);
           
           %transform array Y so values are 1-5 (same ass classes)   
           for m = 1 : size(Y, 1)
        
                if Y(m) < 1            
                    Y(m) = 1;            
                elseif Y(m) > 5            
                    Y(m) = 5;            
                end
        
           end
           
           %error matrix  
           error_matrix = confusionmat(chk_data(:, end), Y);
           
           %overall accuracy
           N = size(chk_data, 1);
           OA_arr(i) = sum(diag(error_matrix))/N;

           %producer's accuracy for each class
           PA_1_arr(i) = error_matrix(1, 1)/(sum(error_matrix(1, :)));
           PA_2_arr(i) = error_matrix(2, 2)/(sum(error_matrix(2, :)));
           PA_3_arr(i) = error_matrix(3, 3)/(sum(error_matrix(3, :))); 
           PA_4_arr(i) = error_matrix(4, 4)/(sum(error_matrix(4, :)));
           PA_5_arr(i) = error_matrix(5, 5)/(sum(error_matrix(5, :)));
          
           %user's accuracy for each class
           UA_1_arr(i) = error_matrix(1, 1)/(sum(error_matrix(:, 1)));
           UA_2_arr(i) = error_matrix(2, 2)/(sum(error_matrix(:, 2)));
           UA_3_arr(i) = error_matrix(3, 3)/(sum(error_matrix(:, 3)));
           UA_4_arr(i) = error_matrix(4, 4)/(sum(error_matrix(:, 4)));
           UA_5_arr(i) = error_matrix(5, 5)/(sum(error_matrix(:, 5)));

           %K            
           sum_1 = 0;           
           for j = 1:5
               sum_1 = sum_1+sum(error_matrix(:, j))*sum(error_matrix(j, :));
           end

           K_arr(i) = (N*sum(diag(error_matrix))-sum_1)/(N^2-sum_1);
           
end

%calculate and save mean average of each metric
sum_1 = 0;
sum_2 = 0;
sum_3 = 0;
sum_4 = 0;
sum_5 = 0;

for i = 1 : 5
    sum_1 = sum_1+PA_1_arr(i);
    sum_2 = sum_2+PA_2_arr(i);
    sum_3 = sum_3+PA_3_arr(i);
    sum_4 = sum_4+PA_4_arr(i);
    sum_5 = sum_5+PA_5_arr(i);
end

PA_1 = sum_1/5;
PA_2 = sum_2/5;
PA_3 = sum_3/5;
PA_4 = sum_4/5;
PA_5 = sum_5/5;

sum_1 = 0;
sum_2 = 0;
sum_3 = 0;
sum_4 = 0;
sum_5 = 0;

for i = 1 : 5
    sum_1 = sum_1+UA_1_arr(i);
    sum_2 = sum_2+UA_2_arr(i);
    sum_3 = sum_3+UA_3_arr(i);
    sum_4 = sum_4+UA_4_arr(i);
    sum_5 = sum_5+UA_5_arr(i);
end

UA_1 = sum_1/5;
UA_2 = sum_2/5;
UA_3 = sum_3/5;
UA_4 = sum_4/5;
UA_5 = sum_5/5;

sum_1 = 0;
sum_2 = 0;

for i = 1 : 5
    sum_1 = sum_1+OA_arr(i);
    sum_2 = sum_2+K_arr(i);
end

OA = sum_1/5;
K = sum_2/5;

%array needed for plots
y = zeros(size(Y, 1), 1);
for i = 1 : size(Y, 1)
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
scatter(y,Y); grid on;
xlabel('Data'); ylabel('Predicted Values');
title('Predicted Values');

%real values
figure();
scatter(y, chk_data(:, end)); grid on;
xlabel('data'); ylabel('Real Values');
title('Real Values');

%learning curve and error in relation to the number of iterations
figure();
grid on;
plot([trn_error val_error]);
xlabel('Iterations');
ylabel('Error');
legend('Training error', 'Validation error');
t = "Learning Curve";
title(t);