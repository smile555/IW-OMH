%% init the workspace
close all; clear; clc; warning off;

%% load dataset
train_param.current_bits=16;
train_param.current_a=0;  %the ratio of unpair data
train_param.current_b=0.5;  %the ratio of unpair image data in all unpair data
train_param.max_iter=5;
train_param.ds_name='MIRFLICKR'; %MIRFLICKR  NUSWIDE21
train_param.train_unpair=true;
train_param.query_unpair=false;
[train_param,XTrain,LTrain,XQuery,LQuery,Vector] = load_dataset(train_param);


%% IW-MOH
train_param.current_hashmethod='W';
OURparam=train_param;
OURparam.alpha= 100;
OURparam.theta = 1;
OURparam.delta = 1;
[e,~]=evaluate_OMHIW(XTrain,LTrain,XQuery,LQuery,Vector,OURparam);
