function [trn_data,val_data,chk_data] = split(data)

    idx=randperm(length(data));
    trnIdx=idx(1:round(length(idx)*0.6));
    chkIdx=idx(round(length(idx)*0.6)+1:round(length(idx)*0.8));
    tstIdx=idx(round(length(idx)*0.8)+1:end);
    trnX=data(trnIdx,1:end-1);
    chkX=data(chkIdx,1:end-1);
    tstX=data(tstIdx,1:end-1);
    
    xmin=min(trnX,[],1);
    xmax=max(trnX,[],1);
    trnX=(trnX-repmat(xmin,[length(trnX) 1]))./(repmat(xmax,[length(trnX) 1])-repmat(xmin,[length(trnX) 1]));
    chkX=(chkX-repmat(xmin,[length(chkX) 1]))./(repmat(xmax,[length(chkX) 1])-repmat(xmin,[length(chkX) 1]));
    tstX=(tstX-repmat(xmin,[length(tstX) 1]))./(repmat(xmax,[length(tstX) 1])-repmat(xmin,[length(tstX) 1]));
    
    trn_data=[trnX data(trnIdx,end)];
    val_data=[chkX data(chkIdx,end)];
    chk_data=[tstX data(tstIdx,end)];    