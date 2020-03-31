%% L?s in ljudfiler fr?n en katalog

%datafolder = '/Users/kalle/Documents/projekt/filkand_2020_fosterdiagnostik/dataset1';
%datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset1';
%datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset2';
datafolder = 'C:\Users\Johan\OneDrive\Skrivbord\Dataset_ljudfiler';

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
        
        % R?kna ut vilket som ?r fram?tkanalen?
        % Hur d??
        % F?rsta f?rs?ket. Gissa att den med mest energi
        % ?r fram?tkanalen. 
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
            % provar att hoppa ?ver f?rsta och sista
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
            end;
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
for k = 1:size(pos,4);
    figure(9); 
    subplot(1,2,1);
    colormap(gray);
    imagesc(pos(:,:,:,k));
    subplot(1,2,2);
    colormap(gray);
    imagesc(reshape(m+u(:,1:KK)*s(1:KK,1:KK)*v(k,1:KK)',50,150));
    title(num2str(k));
    pause; 

end

%%
KK = 5;
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
    title(['Reprojection using ' num2str(k) ' compponents.']);
    pause(0.1);
end


