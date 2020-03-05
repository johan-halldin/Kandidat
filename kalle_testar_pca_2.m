%% Läs in ljudfiler från en katalog

%datafolder = '/Users/kalle/Documents/projekt/filkand_2020_fosterdiagnostik/dataset1';
datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset1';
datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset2';

a = dir(fullfile(datafolder,'*.wav'));
%%
if 1,
    clear facit
    % save facit1 facit
else
    load facit3
end
%%
if 1,
    for i = 1:length(a);
        [y,fs]=audioread(fullfile(datafolder,a(i).name));
        
        % Räkna ut vilket som är framåtkanalen?
        % Hur då?
        % Första försöket. Gissa att den med mest energi
        % är framåtkanalen.
        energi = [norm(y(:,1)) norm(y(:,2))];
        facit(i).energi = energi;
        [sortv,sorti]=sort(-energi);
        facit(i).kanal = sorti;
        
        
        %     figure(1);
        %     plot(y(:,1));
        %     figure(2);
        %     plot(y(:,2));
        
        ytmp = y(:,facit(i).kanal(1));
        
        %s = spectrogram(y(:,1));
        %spectrogram(y,kaiser(128,18),120,128,1E3,'yaxis');
        %title('Quadratic Chirp: start at 100Hz and cross 200Hz at t=1sec');
        
        
        
        s=spectrogram(ytmp,1024,1008,1024);
        figure(1); clf; subplot(2,1,1); hold off; imagesc(abs(s(1:50,:)));
        colormap(gray);
        axis xy
        
        x = abs(s(1:50,:));
        x = x.*(((1:50)')*ones(1,size(x,2)));
        figure(1); subplot(2,1,2); hold off; plot(sum(x));
        z = sum(x);
        xx = -1000:1000;
        ss = 300;
        gg = 1/sqrt(2*pi*ss^2)*exp( -(xx).^2/(2*ss^2) );
        hh = conv2(z,gg,'same');
        ii = 2:(length(hh)-1);
        lokmax = 1+ find( (hh(ii)>hh(ii-1))  & (hh(ii)>hh(ii+1)) & (hh(ii)>2) );
        figure(1); subplot(2,1,2); hold on; plot(hh,'LineWidth',3);
        
        figure(1); subplot(2,1,1); hold on;
        plot([lokmax],10*ones(size(lokmax)),'y*');
        
        facit(i).lokmax = lokmax;
        
        %         if facit(i).class == 1,
        %             facit(i).lokmax = lokmax;
        %         else
        %             facit(i).lokmax = [];
        %         end
        pause(0.1);
        
    end;
    
end;

%%

for i = 1:length(a);
    %     figure(1);
    %     clf;
    %     plot(diff(facit(i).lokmax),'*');
    %     axis([0 20 0 2000]);
    %     pause
    facit(i).period = median(diff(facit(i).lokmax));
end;

%%

pos = zeros(50,150,1,0);
%neg = zeros(32,32,1,0);

for i = 1:length(a);
    [y,fs]=audioread(fullfile(datafolder,a(i).name));
    y = y(:,facit(i).kanal(1));
    s=spectrogram(y,1024,1008,1024);
    x = abs(s(1:50,:));
    
    w2 = round(facit(i).period*0.6);
    w = w2*2;
    %for mid = (w/2+10):(length(x)-w/2-10),
    %end
    if 1,
        for mid = facit(i).lokmax(2:(end-1)),
            % provar att hoppa över första och sista
            if (mid>(w2+10)) & (mid< (length(x)-w2-10)),
                cutout = x(1:50,(mid-w2):(mid+w2-1));
                cutout = conv2(cutout,ones(1,20)/20,'same');
                cutout = cutout(:,10:20:end);
                cutout = cutout/max(cutout(:));
                cutout = imresize(cutout,[50 150]);
                figure(3); clf;
                imagesc(cutout);
                title([num2str(i) ' - ' num2str(mid)]);
                pause(0.1);
                pos = cat(4,pos,cutout);
            end
        end
    end
end

%%

if 1,
    pos_smooth = pos;
    for k = 1:size(pos_smooth,4);
        pos_smooth(:,:,:,k)=conv2(pos_smooth(:,:,:,k),ones(5,5)/25,'same');
    end
end
%%

figure(4);
subplot(1,1,1)
colormap(gray);
montage(pos);

%% Do the PCA

% Reshape data to matrix form

M = reshape(pos(:),50*150,size(pos,4));
M_smooth = reshape(pos_smooth(:),50*150,size(pos,4));

% Calculate mean
m = mean(M,2);

% Remove mean

M2 = M - m*ones(1,size(M,2));
M2_smooth = M_smooth - m*ones(1,size(M,2));

%
[u,s,v]=svd(M2,0);

%%
figure(5);
plot(diag(s),'*');

%%
figure(6);
subplot(4,4,1);
colormap(gray);
imagesc(reshape(m,50,150));
for k = 1:15;
    subplot(4,4,k+1);
    colormap(gray);
    imagesc(reshape(u(:,k),50,150));
end

v_smooth = (inv(s(1:3,1:3))*u(:,1:3)'*M2_smooth)';
v_smooth = (inv(s(1:100,1:100))*u(:,1:100)'*M2_smooth)';

%%
figure(7);
plot(v(:,2),v(:,3),'*');
figure(8);
hold off;
plot3(v(:,1),v(:,2),v(:,3),'b*');
hold on;
plot3(v_smooth(:,1),v_smooth(:,2),v_smooth(:,3),'r.');

%%
KK = 5;
%for k = 1:size(pos,4);
for k = 50:50:500;
    figure(9);
    subplot(1,2,1);
    colormap(gray);
    imagesc(pos(:,:,:,k));
    title('Original');
    subplot(1,2,2);
    colormap(gray);
    imagesc(reshape(m+u(:,1:KK)*s(1:KK,1:KK)*v(k,1:KK)',50,150));
    title(['Reprojection using ' num2str(KK) ' compponents.']);
    pause;
end

%%
KK = 1;
%for k = 1:size(pos,4);
for k = 50:50:500;
    figure(9);
    subplot(1,2,1);
    colormap(gray);
    imagesc(pos_smooth(:,:,:,k));
    title('Original');
    subplot(1,2,2);
    colormap(gray);
    imagesc(reshape(m+u(:,1:KK)*s(1:KK,1:KK)*v_smooth(k,1:KK)',50,150));
    title(['Reprojection using ' num2str(KK) ' compponents.']);
    pause;
end

%%
for k = 1:100;
    figure(7);
    subplot(1,2,1);
    plot(v(:,2),v(:,3),'*');
    [v1,v2]=ginput(1);
    subplot(1,2,2);
    tmp = m+u(:,1)*s(1,1)*v1 + u(:,2)*s(2,2)*v2;
    tmp = reshape(tmp,50,150);
    colormap(gray);
    imagesc(tmp);
end

%%

figure(8);
hold off;
plot3(v(:,1),v(:,2),v(:,3),'b*');
hold on;
plot3(v_smooth(:,1),v_smooth(:,2),v_smooth(:,3),'r.');

%%
plotst = {...
    'b*'   ...
    'g*'   ...
    'r*'   ...
    'c*'   ...
    'm*'   ...
    'bo'   ...
    'go'   ...
    'ro'   ...
    'co'   ...
    'mo'};

n_clusters = 10;
[idx,c] = kmeans(v(:,1:3), n_clusters);
figure(8); clf;
hold on;
for i = 1:n_clusters;
    tmp = find(idx==i);
    hold on;
    plot3(v(tmp,1),v(tmp,2),v(tmp,3),plotst{i});
    plot3(c(i,1),c(i,2),c(i,3),plotst{i},'LineWidth',5,'MarkerSize',10);
end

%%

for i = 1:n_clusters;
    tmp = find(idx==i);
    Mtmp = M(:,tmp);
    mtmp = mean(Mtmp,2);
    Mtmp = Mtmp - repmat(mtmp,1,size(Mtmp,2));
    [utmp,stmp,vtmp]=svd(Mtmp,0);
    
    KK = 6;
    for k = 1:length(tmp);
        figure(9);
        subplot(1,2,1);
        colormap(gray);
        imagesc(pos(:,:,:,tmp(k)));
        title('Original');
        subplot(1,2,2);
        colormap(gray);
        imagesc(reshape(mtmp+utmp(:,1:KK)*stmp(1:KK,1:KK)*vtmp(k,1:KK)',50,150));
        title(['Cluster: ' num2str(i) 'Example: ' num2str(k) ' of ' num2str(length(tmp)) ' Reprojection using ' num2str(KK) ' compponents.']);
        pause;
    end
    
end


%%


for i = 1:n_clusters;
    tmp = find(idx==i);
    Mtmp = M(:,tmp);
    mtmp = mean(Mtmp,2);
    Mtmp = Mtmp - repmat(mtmp,1,size(Mtmp,2));
    [utmp,stmp,vtmp]=svd(Mtmp,0);
    
    figure(1);
    clf;
    plot3(vtmp(:,1),vtmp(:,2),vtmp(:,3),'*');
    pause;
end

%%

%%

for i = 1:n_clusters;
    tmp = find(idx==i);
    Mtmp = M(:,tmp);
    mtmp = mean(Mtmp,2);
    Mtmp = Mtmp - repmat(mtmp,1,size(Mtmp,2));
    [utmp,stmp,vtmp]=svd(Mtmp,0);
    
    KK = 6;
    for k = 1:KK
        figure(9);
        subplot(1,3,1);
        colormap(gray);
        imagesc(reshape(mtmp,50,150));
        title('Mean');
        subplot(1,3,2);
        colormap(gray);
        imagesc(reshape(mtmp+utmp(:,k)*stmp(k,k)*0.1,50,150));
        title(['Cluster: ' num2str(i) 'Mode: ' num2str(k) ' of ' num2str(KK)]);
        subplot(1,3,3);
        colormap(gray);
        imagesc(reshape(mtmp-utmp(:,k)*stmp(k,k)*0.1,50,150));
        title(['Cluster: ' num2str(i) 'Mode: ' num2str(k) ' of ' num2str(KK)]);
        pause;
    end
    
end


%%

for i = 1:n_clusters;
    for i = [7 2 5 8 3];
        tmp = find(idx==i);
        Mtmp = M(:,tmp);
        mtmp = mean(Mtmp,2);
        Mtmp = Mtmp - repmat(mtmp,1,size(Mtmp,2));
        [utmp,stmp,vtmp]=svd(Mtmp,0);
        
        KK = 6;
        for k = 1:KK
            figure(9);
            subplot(1,2,1);
            colormap(gray);
            imagesc(reshape(mtmp,50,150));
            title('Mean');
            for th = linspace(0,2*pi,100);
                subplot(1,2,2);
                colormap(gray);
                imagesc(reshape(mtmp+utmp(:,k)*stmp(k,k)*0.1*cos(th),50,150));
                title(['Cluster: ' num2str(i) 'Mode: ' num2str(k) ' of ' num2str(KK)]);
                pause(0.02);
            end;
            pause;
        end
        
    end
end;

%%

layers = [
    imageInputLayer([61 61 1])
    
    convolution2dLayer(3,10,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,20,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,100,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(7,300,'Padding',0)
    %    reluLayer
    
    transposedConv2dLayer(7,300,'Stride',1,'Cropping',0)
    reluLayer
    
    transposedConv2dLayer(3,100,'Stride',2,'Cropping',0)
    reluLayer
    
    transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
    reluLayer
    
    transposedConv2dLayer(3,20,'Stride',2,'Cropping',1)
    reluLayer
    
    %     transposedConv2dLayer(3,20,'Stride',2,'Cropping',0)
    %     reluLayer
    %
    %     transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
    %     reluLayer
    %
    %     transposedConv2dLayer(3,20,'Stride',2,'Cropping',1)
    %     reluLayer
    %
    convolution2dLayer(1,20,'Padding',0)
    reluLayer
    convolution2dLayer(1,20,'Padding',0)
    reluLayer
    convolution2dLayer(1,1,'Padding',0)
    
    regressionLayer];

%%

% layers = [
%     imageInputLayer([50 150 1])
%     convolution2dLayer(3,10,'Padding',0)
%     reluLayer
%     maxPooling2dLayer(3,'Stride',2)
%     convolution2dLayer(3,20,'Padding',0)
%     reluLayer
%     maxPooling2dLayer(3,'Stride',2)
%     convolution2dLayer(3,30,'Padding',0)
%     reluLayer
%     maxPooling2dLayer(3,'Stride',2)
%     convolution2dLayer([3 16],40,'Padding',0)
%     reluLayer
%     transposedConv2dLayer([3 16],40,'Stride',1,'Cropping',0)
%     reluLayer
%     transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
%     reluLayer
%     transposedConv2dLayer(3,40,'Stride',1,'Cropping',0)
%     reluLayer
%     transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
%     reluLayer
%     transposedConv2dLayer(3,40,'Stride',1,'Cropping',0)
%     reluLayer
%     transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
%     reluLayer
%     transposedConv2dLayer([8 4],1,'Stride',1,'Cropping',0)
%     reluLayer
%     regressionLayer];

layers = [
    imageInputLayer([50 150 1])
    convolution2dLayer([3 3],10,'Padding',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer(3,20,'Padding',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer(3,30,'Padding',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer([3 16],40,'Padding',0)
    reluLayer
    transposedConv2dLayer([3 16],40,'Stride',1,'Cropping',0)
    reluLayer
    transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
    reluLayer
    transposedConv2dLayer(3,40,'Stride',1,'Cropping',0)
    reluLayer
    transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
    reluLayer
    transposedConv2dLayer(3,40,'Stride',1,'Cropping',0)
    reluLayer
    transposedConv2dLayer(3,40,'Stride',2,'Cropping',0)
    reluLayer
    transposedConv2dLayer([8 4],1,'Stride',1,'Cropping',0)
    regressionLayer];

layers = [
    imageInputLayer([50 150 1])
    convolution2dLayer([50 150],10,'Padding',0)
    transposedConv2dLayer([50 150],1,'Stride',1,'Cropping',0)
    regressionLayer];

layers = [
    imageInputLayer([50 150 1])
    convolution2dLayer([3 3],10,'Padding',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer([23 73],10,'Padding',0)
    transposedConv2dLayer([23 73],5,'Stride',1,'Cropping',0)
    reluLayer
    transposedConv2dLayer([6 6],1,'Stride',2,'Cropping',0)
    regressionLayer];

layers = [
    imageInputLayer([50 150 1])
    convolution2dLayer([3 3],10,'Padding',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer([3 3],10,'Padding',0)
    reluLayer
    maxPooling2dLayer(3,'Stride',2)
    convolution2dLayer([10 35],3,'Padding',0)
    transposedConv2dLayer([10 35],10,'Stride',1,'Cropping',0)
    reluLayer
    transposedConv2dLayer([6 6],40,'Stride',2,'Cropping',0)
    reluLayer
    transposedConv2dLayer([4 4],1,'Stride',2,'Cropping',0)
    regressionLayer];

% imageLayer = imageInputLayer([50,150,1]);
%
% encodingLayers = [ ...
%     convolution2dLayer(3,16,'Padding','same'), ...
%     reluLayer, ...
%     maxPooling2dLayer(2,'Padding','same','Stride',2), ...
%     convolution2dLayer(3,8,'Padding','same'), ...
%     reluLayer, ...
%     maxPooling2dLayer(2,'Padding','same','Stride',2), ...
%     convolution2dLayer(3,8,'Padding','same'), ...
%     reluLayer, ...
%     maxPooling2dLayer(2,'Padding','same','Stride',2)];
%
% decodingLayers = [ ...
%     createUpsampleTransponseConvLayer(2,8), ...
%     reluLayer, ...
%     createUpsampleTransponseConvLayer(2,8), ...
%     reluLayer, ...
%     createUpsampleTransponseConvLayer(2,16), ...
%     reluLayer, ...
%     convolution2dLayer(3,1,'Padding','same'), ...
%     clippedReluLayer(1.0), ...
%     regressionLayer];
%
% layers = [imageLayer,encodingLayers,decodingLayers];


% Kanske relu på slutet, kanske inte
% Kör valideringssätt.

% Train
miniBatchSize = 400;
max_epochs = 2000;
learning_rate = 0.01;
options = trainingOptions( 'adam',...
    'MaxEpochs',max_epochs,...
    'MiniBatchSize', miniBatchSize,...
    'InitialLearnRate',learning_rate, ...
    'Plots', 'training-progress', ...
    'ValidationData',{pos,pos});

if 1, %Träna inte denna gång utan använd ett nät som är sparat
    net = trainNetwork(pos_smooth, pos_smooth, layers, options);
else
    load trained_auto_net8758
end,


%
% predict
[posr] = predict(net,pos(:,:,:,:));

figure(12);
montage(pos(:,:,:,1:36));
figure(13);
montage(posr(:,:,:,1:36));
figure(11);
subplot(1,2,1); colormap(gray); imagesc(pos(:,:,:,1));
subplot(1,2,2); colormap(gray); imagesc(posr(:,:,:,1));


%%
KK = 3;
%for k = 1:size(pos,4);
for k = 10:10:500;
    figure(9);
    subplot(1,3,1);
    colormap(gray);
    imagesc(pos(:,:,:,k));
    title('Original');
    subplot(1,3,2);
    colormap(gray);
    imagesc(reshape(m+u(:,1:KK)*s(1:KK,1:KK)*v(k,1:KK)',50,150));
    title(['Reprojection using pca ' num2str(KK) ' components.']);
    subplot(1,3,3);
    colormap(gray);
    imagesc(posr(:,:,:,k));
    title(['Reprojection using autoencoders ' num2str(KK) ' components.']);
    pause;
end



%%

layers2 = net.Layers

layers_ = [
    layers(1:12)
    fullyConnectedLayer(numResponses)
    regressionLayer];



compressed = forward(encoderNet, x);



tmp = saveobj(net);
tmp.Layers = tmp.Layers([1:8 14]);
encoder = SeriesNetwork.loadobj(tmp)
blubb = predict(encoder,pos(:,:,:,1))

tmp = saveobj(net);
tmp.Layers = [imageInputLayer([1 1 3],'Normalization','none');tmp.Layers([9:14])];
decoder = SeriesNetwork.loadobj(tmp)
blubb2 = predict(decoder,reshape(blubb(1,:),[1 1 3]))

%%

v2 = predict(encoder,pos(:,:,:,:))
figure(8);
hold off;
plot3(v2(:,1),v2(:,2),v2(:,3),'b*');
hold on;
plot3(v_smooth(:,1),v_smooth(:,2),v_smooth(:,3),'r.');

%%
mm = mean(v2);
ss = std(v2);
ee = eye(3);
figure(11);
for k = 1:3;
    subplot(1,2,1);
    colormap(gray);
    meanimage = predict(decoder,reshape(mm,[1 1 3]));
    imagesc(meanimage);
    for th = linspace(0,2*pi,100);
        subplot(1,2,2);
        colormap(gray);
        meanimage = predict(decoder,reshape(mm+cos(th)*ss.*ee(k,:),[1 1 3]));
        imagesc(meanimage);
        title(['Mode: ' num2str(k) ' of ' num2str(3)]);
        pause(0.05);
    end;
    pause;
end;
