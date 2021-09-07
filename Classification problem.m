%%Coursework of <name> <knumber>, Feb 2021%%
% rename this file to k12345678.m for submission, using your k number
%%%%%%%%%%%%%
%% initialization

clear; close all; clc; format longg;

load 'USPS_dataset9296.mat' X t; % loads 9296 handwritten 16x16 images X dim(X)=[9296x256] and the lables t in [0:9] dim(t)=[9298x1]
[Ntot,D] =      size(X);         % Ntot = number of total dataset samples. D =256=input dimension

% Anonymous functions as inlines
show_vec_as_image16x16 =    @(row_vec)      imshow(-(reshape(row_vec,16,16)).');    % shows the image of a row vector with 256 elements. For matching purposes, a negation and rotation are needed.
sigmoid =                   @(x)            1./(1+exp(-x));                         % overwrites the existing sigmoid, in order to avoid the toolbox
LSsolver =                  @(Xmat,tvec)    ( Xmat.' * Xmat ) \ Xmat.' * tvec;      % Least Square solver

PLOT_DATASET =  0;      % For visualization. Familiarise yourself by running with 1. When submiting the file, set back to 0
if PLOT_DATASET
    figure(8); sgtitle('First 24 samples and labels from USPS data set');
    for n=1:4*6
        subplot(4,6,n);
        show_vec_as_image16x16(X(n,:));
        title(['t_{',num2str(n),'}=',num2str(t(n)),'   x_{',num2str(n),'}=']);
    end
end

% code here initialization code that manipulations the data sets

load USPS_dataset9296
over0=[];
over1=[];
row0=1;
row1=1;
for ind=1:length(t)
    if t(ind)==0
        over0(row0,:)=X(ind,:);
        row0=row0+1;
    elseif t(ind)==1
        over1(row1,:)=X(ind,:);
        row1=row1+1;
    end
end       
ori11=size(over1);
ori00=size(over0);
ori1=ori11(1); % original colloum of 0 set
ori0=ori00(1); % original collume of 1 set
numtra0=round(ori0*0.7);
numtra1=round(ori1*0.7);

overtrain=[over0(1:numtra0,:);over1(1:numtra1,:)];

overtrainpic=overtrain;

rowoverori=size(overtrain);
overtrain=[ones(rowoverori(1),1),overtrain];
numtrain = rowoverori(1);

t=[zeros(1,numtra0),ones(1,numtra1)];

%%Section 1

% dim(theta)=257:
% supporting code here

thERM=LSsolver(overtrain,t');
xaxis=1:rowoverori(1);
for l=1:rowoverori(1)
    ul=overtrain(l,:);
    tl(l)=thERM'*ul';
end
figure(1); hold on; title('Section 1: ERM regression, quadratic loss');
% code for plot here
plot(xaxis,tl,'g')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
hold on; 
plot(xaxis,t,'--.k')
hold on

numvali0=ori0-numtra0;
numvali1=ori1-numtra1;
numvali = numvali0+numvali1; 
xvalo=[over0(numtra0+1:ori0,:);over1(numtra1+1:ori1,:)];
xval=[ones(numvali0+numvali1,1),xvalo];
tval=[zeros(1,numvali0),ones(1,numvali1)];

% supporting code here that help to calculate and displaying without a ";" the two variables
% traininglossLS_257    % Training   loss when dim(theta)=257.
traininglossLS_257=1/rowoverori(1)*norm(t'-overtrain*thERM)^2
% validationlossLS_257  % Validation loss when dim(theta)=257.
validationlossLS_257=1/(numvali0+numvali1)*norm(tval'-xval*thERM)^2

% dim(theta)=10:
load USPS_dataset9296
over0F9=[];
over1F9=[];
row0F9=1;
row1F9=1;
for indF9=1:length(t)
    if t(indF9)==0
        over0F9(row0F9,1:9)=X(indF9,1:9);
        row0F9=row0F9+1;
    elseif t(indF9)==1
        over1F9(row1F9,1:9)=X(indF9,1:9);
        row1F9=row1F9+1;
    end
end       
ori11F9=size(over1F9);
ori00F9=size(over0F9);
ori1F9=ori11F9(1); % original colloum of 0 set
ori0F9=ori00F9(1); % original collume of 1 set
numtra0F9=round(ori0F9*0.7);
numtra1F9=round(ori1F9*0.7);

overtrainF9=[over0F9(1:numtra0F9,:);over1F9(1:numtra1F9,:)];

overtrainpicF9=overtrainF9;

rowoveroriF9=size(overtrainF9);
overtrainF9=[ones(rowoveroriF9(1),1),overtrainF9];
numtrainF9 = rowoveroriF9(1);

t=[zeros(1,numtra0F9),ones(1,numtra1F9)];


thERMF9=LSsolver(overtrainF9,t');
xaxisF9=1:rowoveroriF9(1);
for lF9=1:rowoveroriF9(1)
    ul=overtrainF9(lF9,:);
    tlF9(lF9)=thERMF9'*ul';
end


numvali0F9=ori0F9-numtra0F9;
numvali1F9=ori1F9-numtra1F9;
xvaloF9=[over0F9(numtra0F9+1:ori0F9,:);over1F9(numtra1F9+1:ori1F9,:)];
xvalF9=[ones(numvali0F9+numvali1F9,1),xvaloF9];
tvalF9=[zeros(1,numvali0F9),ones(1,numvali1F9)];





% code for plot here (using the "hold on", it will overlay)
plot(xaxisF9,tlF9,'r')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
hold on; 
% supporting code here that help to calculate and displaying without a ";" the two variables
% traininglossLS_10    % Training   loss when dim(theta)=10.
traininglossLS_10=1/rowoveroriF9(1)*norm(t'-overtrainF9*thERMF9)^2
% validationlossLS_10  % Validation loss when dim(theta)=10.
validationlossLS_10=1/(numvali0F9+numvali1F9)*norm(tvalF9'-xvalF9*thERMF9)^2


% complete the insight:
display('The predictions with the longer and shorter feature vectors are different because the longer feature vectors provide more dimensions so that we could gain more information regarding the model from many aspects and a good learning result could be obtained. ')

%%Section 2

% supporting code here
I=100; %number of iterations
S=32; %mini-batch size
gamma=0.05; %learning rate
th=thERM; %initialization

clmargtr=(2*t-1)'.*(th'*overtrain')'; 
trloss(1)=1/numtrain*sum(log(1+exp(-clmargtr))); 
clmargval=(2*tval-1)'.*(th'*xval')'; %classification margins on the validation set
valloss(1)=1/numvali*sum(log(1+exp(-clmargval))); %validation hinge-at-zero loss
for i=1:I
    
    ind=mnrnd(1,1/numtrain*ones(numtrain,1),S)*[1:numtrain]'; %generate S random indices
    
    g=zeros(257,1);

    for s=1:S
        mm=1/S*(1/(1+exp(-th'*overtrain(ind(s),:)'))-t(ind(s)))*overtrain(ind(s),:)'; 
        g=g+mm;


    end
    th=th-gamma*g;
    
    %classification margin
    
    clmargtr=(2*t-1)'.*(th'*overtrain')'; %classification margins on the training set
    trloss(i)=1/numtrain*sum(log(1+exp(-clmargtr))); %training logistic regression loss
    clmargval=(2*tval-1)'.*(th'*xval')'; %classification margins on the validation set
    valloss(i)=1/numvali*sum(log(1+exp(-clmargval))); %validation logistic regression loss
end
figure(2); hold on; title('Section 2: Logistic Regression');
% code for plot here
plot([1:I],trloss,'k','LineWidth',2); hold on; plot([1:I],valloss,'r','LineWidth',2);
xlabel('iterations','Interpreter','latex','FontSize',12)
legend('training','validation')
ylabel('loss');

% complete the insight:
display('I have chosen S=<64> and gamma=<0.05> because through the calculation 1974/50=39.48, if we choose S<39.48, there must some lost sample points that cannot be taken into iteration. Apart from that, an S that is far too small might make the training process not that smooth, which could be observed in figure(2). On the other hand, an S with huge value would increase training time. As for the value of gamma, if we set gamma a large value, the iteration would be likely to miss the global minimi and "jump" forward and backwards within an interval while a small gamma may lower the speed of decreasing training loss. Therefore, after several times running, I chose the value as above.');

%%Section 3

N=size(overtrainpic,1);
[W,D]=eig(1/N*overtrainpic'*overtrainpic);
w1=W(:,1); w2=W(:,2); w3=W(:,3);

% some code here, and replace the three <???> in the plots:
figure(3); sgtitle('Section 3: PCA most significant Eigen vectors');
subplot(2,2,1);
imv1=reshape(w1,[16,16])';
show_vec_as_image16x16(imv1); 
imagesc(imv1); 
colormap(gray);
title('Most significant');

subplot(2,2,2); 
imv2=reshape(w2,[16,16])';
show_vec_as_image16x16(imv2);
imagesc(imv2); 
colormap(gray);
title('Second significant');

subplot(2,2,3); 
imv3=reshape(w3,[16,16])';
show_vec_as_image16x16(imv3);
imagesc(imv3); 
colormap(gray);
title('Third significant');


figure(4); sgtitle('Section 3: Estimating using PCA, M = number of significant components');
% some code here, and replace the three <???> in the plots:
z(1)=w1'*overtrainpic(1,:)'; z(2)=w2'*overtrainpic(1,:)';z(3)=w3'*overtrainpic(1,:)'; %	Contributions of 3 principal components

subplot(2,2,1); 
im1=reshape(overtrainpic(1,:),[16,16])';
show_vec_as_image16x16(im1);   
imagesc(im1); colormap(gray);
title('First training set image');

subplot(2,2,2); 
im1reconstv=z(1)*w1; 
im1reconst=reshape(im1reconstv,[16,16])';
show_vec_as_image16x16(im1reconst);   
imagesc(im1reconst); colormap(gray);
title('Reconstracting using M=1 most significant components');

subplot(2,2,3); 
im2reconstv=z(1)*w1+z(2)*w2; 
im2reconst=reshape(im2reconstv,[16,16])';
show_vec_as_image16x16(im2reconst);
imagesc(im2reconst); colormap(gray);
title('Reconstracting using M=2');

subplot(2,2,4); 
im3reconstv=z(1)*w1+z(2)*w2+z(3)*w3; 
im3reconst=reshape(im3reconstv,[16,16])';
show_vec_as_image16x16(im3reconstv);
imagesc(im3reconst); colormap(gray);
title('Reconstracting using M=3');

figure(5); title('Significant PCA components over all training set');
% code for plot3 here

z0(1,:)=w1'*overtrainpic(1:1086,:)';
z0(2,:)=w2'*overtrainpic(1:1086,:)';
z00=z0;
z0(3,:)=w3'*overtrainpic(1:1086,:)';


z1(1,:)=w1'*overtrainpic(1087:1974,:)';
z1(2,:)=w2'*overtrainpic(1087:1974,:)';
z11=z1;
z1(3,:)=w3'*overtrainpic(1087:1974,:)';

plot3(z0(1,:),z0(2,:),z0(3,:),'o','MarkerSize',8);
hold on;
plot3(z1(1,:),z1(2,:),z1(3,:),'x','MarkerSize',8);
xlabel('$z_1$','Interpreter','latex','FontSize',16);
ylabel('$z_2$','Interpreter','latex','FontSize',16);
zlabel('$z_3$','Interpreter','latex','FontSize',16);
grid on;

legend('digit 2','digit 9');


%%Section 4
% supporting code here
zover=[z00';z11'];
zover=[ones(1974,1),zover];
I=100; %number of iterations
%S=5; 
%gamma=0.05; 
th4=[1,1,1]'; %initialization

clmargtr=(2*t-1)'.*(th4'*zover')'; 
trloss(1)=1/numtrain*sum(log(1+exp(-clmargtr))); 

for i=1:I
    
    ind=mnrnd(1,1/numtrain*ones(numtrain,1),S)*[1:numtrain]'; %generate S random indices
    g=zeros(3,1);

    for s=1:S
        mm=1/S*(1/(1+exp(-th4'*zover(ind(s),:)'))-t(ind(s)))*zover(ind(s),:)'; 
        g=g+mm;
    end
    th4=th4-gamma*g;
    
    %classification margin
    clmargtr=(2*t-1)'.*(th4'*zover')'; %classification margins on the training set
    trloss(i)=1/numtrain*sum(log(1+exp(-clmargtr))); 
end
figure(6); hold on; title('Section 4: Logistic Regression');
% code for plot here
plot([1:I],trloss,'k','LineWidth',2); 
xlabel('iterations','Interpreter','latex','FontSize',12)
legend('training')
ylabel('loss');
% complete the insight:
display('Comparing with the solution in Section 2, I conclude that the two sections provide similar training loss after 50 iterations as the first three principal components contain most of the valid information of the origin data set. Hence, PCA is a good way of saving a large amount of information by using lower dimensions of data. ');




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A list of functions you may want to familiarise yourself with. Check the help for full details.

% + - * / ^ 						% basic operators
% : 								% array indexing
% * 								% matrix mult
% .* .^ ./							% element-wise operators
% x.' (or x' when x is surely real)	% transpose
% [A;B] [A,B] 						% array concatenation
% A(row,:) 							% array slicing
% round()
% exp()
% log()
% svd()
% max()
% sqrt()
% sum()
% ones()
% zeros()
% length()
% randn()
% randperm()
% figure()
% plot()
% plot3()
% title()
% legend()
% xlabel(),ylabel(),zlabel()
% hold on;
% grid minor;