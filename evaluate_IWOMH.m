function [eva,train_time_round] = evaluate_IWOMH(XTrain,LTrain,XQuery,LQuery,Vector,OURparam)
    eva=zeros(1,OURparam.nchunks);
    train_time_round=zeros(1,OURparam.nchunks);
    
    %% Learn Bc
    alpha=OURparam.alpha;
    c=size(LTrain{1,1},2);
    nbits=OURparam.current_bits;

    B_c = sign(randn(nbits,c)); 
    B_c(B_c==0) = -1;
    
    Vector = Vector ./ sum(Vector.^2,2).^0.5;
    
    Vector = Vector';
    
    for i=1:OURparam.max_iter
        W=pinv(B_c*B_c'+alpha)*B_c*Vector';
        
        Q=W*Vector;
        
        for row=1:nbits
            B_c(row,:)=sign(Q(row,:)'-B_c(setdiff(1:nbits,row),:)'*W(setdiff(1:nbits,row),:)*W(row,:)')';        
        end
    end
    
    %% train
    for chunki = 1:OURparam.nchunks
        fprintf('-----chunk----- %3d\n', chunki);
        
        XTrain_new = XTrain{chunki,:};
        LTrain_new = LTrain{chunki,:};
        
        % Hash code learning
        if chunki == 1
            [BB,WW,PP,UU,t] = train_IWOMH0(XTrain_new',LTrain_new',B_c,OURparam);
        else
            [BB,WW,PP,UU,t] = train_IWOMH(XTrain_new',LTrain_new',BB,PP,B_c,OURparam);
        end
        train_time_round(1,chunki) = t;
        fprintf('the %i chunk finished, train time is %d (s)\n',chunki,train_time_round(1,chunki));
        
        %% test
        fprintf('test beginning\n');
            
        h1=ones(1,OURparam.current_bits)*abs(UU{1,1}'*XQuery(:,1:OURparam.image_feature_size)');
        h2=ones(1,OURparam.current_bits)*abs(UU{2,1}'*XQuery(:,OURparam.image_feature_size+1:end)');
        
        % Inv calculation may be appear nan, for more precise results, please use the pinv in train code to calculate U.
        h1(isnan(h1))=10;  
        h2(isnan(h2))=10;
        
        h=max(max(h1),max(h2));
        PI1=h-h1;
        PI2=h-h2;
        
        XQuery_B = compactbit((PI1.*(WW{1,1}*XQuery(:,1:OURparam.image_feature_size)')+PI2.*(WW{2,1}*XQuery(:,OURparam.image_feature_size+1:end)'))'>0); 
        
        B = cell2mat(BB(1:chunki,:));
        XTrain_B = compactbit(B>0);
        
        %mAP
        DHamm = hammingDist(XQuery_B, XTrain_B);
        [~, orderH] = sort(DHamm, 2);
        eva(1,chunki) = mAP(orderH', cell2mat(LTrain(1:chunki,:)), LQuery);
        
        
        fprintf('the %i chunk : mAP=%d\n', chunki,eva(1,chunki));
    end
end

