close all;
clear all;

%load and normalize data
data = csvread('epileptic_seizure_data.csv', 1, 1);
norm_data = data(:, 1 : end-1);
norm_data = normalize(norm_data);
data = [norm_data(:, 1 : end) data(:, end)];

%evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

%grid
num_features = 4;          
num_ra = 7;

grid_parameters = zeros(num_features, num_ra, 2);

grid_parameters(1, :, 1) = 3;
grid_parameters(2, :, 1) = 6;
grid_parameters(3, :, 1) = 9;
grid_parameters(4, :, 1) = 12;
grid_parameters(:, 1, 2) = 0.2;
grid_parameters(:, 2, 2) = 0.3;
grid_parameters(:, 3, 2) = 0.4;
grid_parameters(:, 4, 2) = 0.5;
grid_parameters(:, 5, 2) = 0.6;
grid_parameters(:, 6, 2) = 0.7;
grid_parameters(:, 7, 2) = 0.8;


%most useful features 
[indexes, weights] = relieff(data(:, 1 : end-1), data(:, end), 6); 

%arrays for OA, number of rules, and selected number of features
OA_cv = zeros(5, 1); 
OA = zeros(num_features,num_ra);
number_of_rules = zeros(num_features,num_ra);
selected_features = zeros(num_features,num_ra);

%grid search
for i = 1 : num_features 
    
    for j = 1 : num_ra 
        
        features = grid_parameters(i, j, 1);
        ra = grid_parameters(i, j, 2);
        
        %split data - 80% train, 20% test
        train_test = cvpartition(data(:, end), 'KFold', 5,'Stratify', true);    
                
        for k = 1:train_test.NumTestSets
           %load train, test data  
           data_80 = data(training(train_test, k), :);
           chk_data = data(test(train_test, k), :);
           
           %75% of training data for training, 25% for evaluation     
           trn_val = cvpartition(data_80(:, end), 'KFold', 4, 'Stratify', true);
           trn_data = data_80(training(trn_val, 2), :);
           val_data = data_80(test(trn_val, 2), :);
           
           %save the most useful features          
           trn_data = [trn_data(:, indexes(1 : features)) trn_data(:, end)];
           val_data = [val_data(:, indexes(1 : features)) val_data(:, end)];
           chk_data = [chk_data(:, indexes(1 : features)) chk_data(:, end)];
           
           %clustering per class         
           [c_1, sig_1] = subclust(trn_data(trn_data(: ,end)==1, :), ra);
           [c_2, sig_2] = subclust(trn_data(trn_data(:, end)==2, :), ra);
           [c_3, sig_3] = subclust(trn_data(trn_data(:, end)==3, :), ra);
           [c_4, sig_4] = subclust(trn_data(trn_data(:, end)==4, :), ra);
           [c_5, sig_5] = subclust(trn_data(trn_data(:, end)==5, :), ra);
           num_rules = size(c_1, 1)+size(c_2, 1)+size(c_3, 1)+size(c_4, 1)+size(c_5, 1);
           
           %FIS          
           fis = newfis('FIS_SC', 'sugeno');
           
           %add Input-Output Variables         
           for m = 1:size(trn_data,2)-1              
               fis = addvar(fis,'input', "in " + m,[0 1]);              
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
           params=[zeros(1, size(c_1, 1)) 0.25*ones(1, size(c_2, 1)) 0.5*ones(1, size(c_3, 1)) 0.75*ones(1, size(c_4, 1)) ones(1, size(c_5, 1))];
           
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
           
           %train and evaluate ANFIS    
           [trn_fis, trn_error, ~, val_fis, val_error] = anfis(trn_data, fis, [100 0 0.01 0.9 1.1], [], val_data);
           Y = evalfis(chk_data(:, 1 : end-1), val_fis);
           Y = round(Y);
           
           %transform array Y so values are 1-5 (same as classes)   
           for m=1:size(Y,1)
        
                if Y(m) < 1            
                    Y(m) = 1;            
                elseif Y(m) > 5           
                    Y(m) = 5;            
                end
        
           end
           
           %error matrix, 5 classes    
           error_matrix = confusionmat(chk_data(:,end),Y);
    
           %overall accuracy    
           N = size(chk_data, 1);
           OA_cv(k) = (error_matrix(1,1) + error_matrix(2,2) + error_matrix(3,3) + error_matrix(4,4) + error_matrix(5,5))/N;
           
        end
        
        %calculate and save mean average of OA 
        OA(i, j) = sum(OA_cv)/5;
        
        %save number of rules and selected number of features        
        number_of_rules(i, j) = size(val_fis.rule, 2);
        selected_features(i, j) = features;
        
    end
    
end

%plot overall accuracy in relation to number of rules
figure();
scatter(reshape(OA, 1, []), reshape(number_of_rules, 1, [])); grid on;
xlabel("Overall Accuracy"); 
ylabel("Number of Rules");
title("Overall Accuracy in relation to number of rules ");

%plot overall accuracy in relation to selected number of features
figure();
scatter(reshape(OA, 1, []), reshape(selected_features, 1, [])); grid on;
xlabel("Overall Accuracy"); 
ylabel("Selected number of features");
title("Overall Accuracy in relation to selected number of features ");