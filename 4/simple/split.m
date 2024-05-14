
function [trn_data, val_data, chk_data] = split(data)
    %splitting data based on their class
    k = 1;
    l = 1;
    for i = 1 : length(data)       
        if data(i, 4) == 1      
            class_1(k) = i;
            k = k+1;
        else
            class_2(l) = i;
            l = l+1;
        end       
    end
    
    %shuffle each class
    indexes_1 = randperm(length(class_1));
    indexes_2 = randperm(length(class_2));
    
    %train data
    dif = (rand - 0.5)*0.1*length(data); %random dif E [-5, 5]% of the dataset length
    
    trn_split_point_1 = round((length(indexes_1)+dif)*0.6); %to ensure the training set has a similar dristibution as the original dataset
    trn_split_point_2 = round((length(indexes_2)-dif)*0.6); 
    
    trn_indexes_1 = indexes_1(1 : trn_split_point_1);
    trn_indexes_2 = indexes_2(1 : trn_split_point_2);
    
    trn_indexes = horzcat(trn_indexes_1, trn_indexes_2); 
    trn_inputs = data(trn_indexes, 1 : end-1);
    
    %validation Data
    dif = (rand - 0.5)*0.1*length(data); %random dif E [-5, 5]% of the dataset length
 
    val_split_point_1 = trn_split_point_1+round((length(indexes_1)+dif)*0.2); %to ensure the training set has a similar dristibution as the original dataset
    val_split_point_2 = trn_split_point_2+round((length(indexes_2)-dif)*0.2); 
    
    val_indexes_1 = indexes_1(trn_split_point_1+1 : val_split_point_1);
    val_indexes_2 = indexes_2(trn_split_point_2+1 : val_split_point_2);
    
    val_indexes = horzcat(val_indexes_1, val_indexes_2);
    val_inputs = data(val_indexes, 1 : end-1);
    
    %test Data   
    chk_indexes_1 = indexes_1(val_split_point_1+1 : end);
    chk_indexes_2 = indexes_2(val_split_point_2+1 : end);
    
    chk_indexes = horzcat(chk_indexes_1, chk_indexes_2);
    chk_inputs = data(chk_indexes, 1 : end-1);
    
    %normalization
    min_x = min(trn_inputs, [], 1);
    max_x = max(trn_inputs, [], 1);
    trn_inputs = (trn_inputs-repmat(min_x, [length(trn_inputs) 1]))./(repmat(max_x, [length(trn_inputs) 1])-repmat(min_x, [length(trn_inputs) 1]));
    val_inputs = (val_inputs-repmat(min_x, [length(val_inputs) 1]))./(repmat(max_x, [length(val_inputs) 1])-repmat(min_x, [length(val_inputs) 1]));
    chk_inputs = (chk_inputs-repmat(min_x, [length(chk_inputs) 1]))./(repmat(max_x, [length(chk_inputs) 1])-repmat(min_x, [length(chk_inputs) 1]));

    %final data
    trn_data=[trn_inputs data(trn_indexes,end)];
    val_data=[val_inputs data(val_indexes,end)];
    chk_data=[chk_inputs data(chk_indexes,end)];

end