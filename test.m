%% Läs in ljudfiler från en katalog

%datafolder = '/Users/kalle/Documents/projekt/filkand_2020_fosterdiagnostik/dataset1';
datafolder = '/Users/kalle/Documents/projekt/filkand_2020_fosterdiagnostik/dataset1';

a = dir(fullfile(datafolder,'*.wav'));
%%
if 1,
    clear facit
    facit(1).kanal = 1;
    facit(2).kanal = 1;
    facit(3).kanal = 2;
    facit(4).kanal = 2;
    facit(5).kanal = 1;
    facit(6).kanal = 1;
    facit(7).kanal = 1;
    facit(1).class = 1;
    facit(2).class = 1;
    facit(3).class = 1;
    facit(4).class = 1;
    facit(5).class = 1;
    facit(6).class = 2;
    facit(7).class = 2;
    % save facit1 facit
else
    load facit3
end
%%
if 1,
    for i = 1:length(a);
        [y,fs]=audioread(fullfile(datafolder,a(i).name));
        
        %     figure(1);
        %     plot(y(:,1));
        %     figure(2);
        %     plot(y(:,2));
        
        y = y(:,facit(i).kanal);

        %s = spectrogram(y(:,1));
        %spectrogram(y,kaiser(128,18),120,128,1E3,'yaxis');
        %title('Quadratic Chirp: start at 100Hz and cross 200Hz at t=1sec');
        
        
        
        s=spectrogram(y,1024,1008,1024);
        figure(1); clf; subplot(2,1,1); hold off; imagesc(abs(s(1:50,:)));
        colormap(gray);
        axis xy
        
        x = abs(s(1:50,:));
        figure(1); subplot(2,1,2); hold off; plot(sum(x));
        z = sum(x);
        xx = -1000:1000;
        ss = 100;
        gg = 1/sqrt(2*pi*ss^2)*exp( -(xx).^2/(2*ss^2) );
        hh = conv2(z,gg,'same');
        ii = 2:(length(hh)-1);
        lokmax = 1+ find( (hh(ii)>hh(ii-1))  & (hh(ii)>hh(ii+1)) & (hh(ii)>2) );
        
        figure(1); subplot(2,1,1); hold on;
        plot([lokmax],10*ones(size(lokmax)),'y*');
 
        facit(i).lokmax = lokmax;

%         if facit(i).class == 1,
%             facit(i).lokmax = lokmax;
%         else
%             facit(i).lokmax = [];
%         end
        pause;
        
    end;
    
end;

%%

pos = zeros(32,32,1,0);
neg = zeros(32,32,1,0);

for i = 1:length(a);
    [y,fs]=audioread(a(i).name);
    y = y(:,facit(i).kanal);
    s=spectrogram(y,1024,1008,1024);
    x = abs(s(1:50,:));
    
    w = 640;
    w2 = w/2;
    %for mid = (w/2+10):(length(x)-w/2-10),
    %end
    if facit(i).class==1,
        for mid = facit(i).lokmax,
            if (mid>(w2+10)) & (mid< (length(x)-w2-10)),
                cutout = x(3:34,(mid-w2):(mid+w2-1));
                cutout = conv2(cutout,ones(1,20)/20,'same');
                cutout = cutout(:,10:20:end);
                cutout = cutout/max(cutout(:));
                %figure(3); clf;
                %imagesc(cutout);
                %title([num2str(i) ' - ' num2str(mid)]);
                %pause;
                pos = cat(4,pos,cutout);
            end;
        end
    else
        for mid = facit(i).lokmax,
            if (mid>(w2+10)) & (mid< (length(x)-w2-10)),
                cutout = x(3:34,(mid-w2):(mid+w2-1));
                cutout = conv2(cutout,ones(1,20),'same');
                cutout = cutout(:,10:20:end);
                cutout = cutout/max(cutout(:));
                %figure(3); clf;
                %imagesc(cutout);
                %title([num2str(i) ' - ' num2str(mid)]);
                %pause;
                neg = cat(4,neg,cutout);
            end;
        end
        
    end
end

%%

figure(4);
subplot(1,2,1)
colormap(gray);
montage(pos);
subplot(1,2,2);
colormap(gray);
montage(neg);

%%

train_im = cat(4,pos,neg);
train_classes = categorical([1*ones(size(pos,4),1);2*ones(size(neg,4),1)]);

% layers = [
%     imageInputLayer([32 32 1], 'Name', 'input')    % Specify input sizes
%     fullyConnectedLayer(2,'Name', 'fully_conn')   % Fully connected is a affine map from 28^2 pixels to 10 numbers
%     softmaxLayer('Name', 'softmax')                % Convert to 'probabilities'
%     classificationLayer('Name', 'classoutput')];   % Specify output layer
layers = [
    imageInputLayer([32 32 1], 'Name', 'input')    % Specify input sizes
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(2,'Name', 'fully_conn')   % Fully connected is a affine map from 28^2 pixels to 10 numbers
    softmaxLayer('Name', 'softmax')                % Convert to 'probabilities'
    classificationLayer('Name', 'classoutput')];   % Specify output layer

miniBatchSize = 500;
max_epochs = 100;           % Specify how long we should optimize
learning_rate = 0.01;     % Try different learning rates
options = trainingOptions( 'sgdm',...
    'MaxEpochs',max_epochs,...
    'MiniBatchSize', miniBatchSize,...
    'InitialLearnRate',learning_rate, ...
    'Plots', 'training-progress');

net = trainNetwork(train_im, train_classes, layers, options);


[Y_result1,scores1] = classify(net,train_im);
accuracy1 = sum(Y_result1 == train_classes)/numel(Y_result1);
disp(['The accuracy on the training set: ' num2str(accuracy1)]);

%%

tmp = net.Layers(2).Weights;
figure(2); colormap(gray);
imagesc(reshape(tmp(1,:),32,32));
imagesc(reshape(tmp(2,:),32,32));

%%

[y,fs]=audioread(fullfile(datafolder,a(4).name));
y = y(:,1);
s=spectrogram(y,1024,1008,1024);
x = abs(s(1:50,:));

x_test = x(3:34,:);
x_test = conv2(x_test,ones(1,20),'same');
x_test = x_test(:,10:20:end);
x_test = x_test/max(x_test(:));
nn = (size(x_test,2)-31);
yres = zeros(1,nn);
ysco = zeros(2,nn);
for j = 1:nn,
    cutout = x_test(:,(j):(j+31));
    [Y_result1,scores1] = classify(net,cutout);
    yres(j)=double(Y_result1);
    ysco(:,j)=double(scores1');
end

figure(1); clf;
imagesc(x_test);
colormap(gray);
axis xy
tmp = find(yres==2)+15;
hold on;
plot(tmp,16*ones(size(tmp)),'*');
