%% Läs in ljudfilsnamn från en katalog

%datafolder = '/Users/kalle/Documents/projekt/filkand_2020_fosterdiagnostik/dataset1';
datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset1';
datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset2';


%Henrik
datafolder = 

a = dir(fullfile(datafolder,'*.wav'));

%% Läs in varje ljudfil och gör inledande analys
% bl a räkna ut vilken kanal som är dominant. 

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
        facit(i).kanal = sorti;  % Här är första talet den dominanta kanalen
        
        
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
        
        facit(i).lokmax = lokmax; % Är är index för topparna
        
        %         if facit(i).class == 1,
        %             facit(i).lokmax = lokmax;
        %         else
        %             facit(i).lokmax = [];
        %         end
        pause(0.1);
        
    end;
    
end;

%% Liten test för att se om avståndet mellan två toppar är ungefär samma
% Använd medianen av avstånden som skattning av storleken.

for i = 1:length(a);
    %     figure(1);
    %     clf;
    %     plot(diff(facit(i).lokmax),'*');
    %     axis([0 20 0 2000]);
    %     pause
    facit(i).period = median(diff(facit(i).lokmax));
end;

%% Här görs urklippen. Jag skapara bilder av storlek 50 x 150 
% för varje topp. Dessa sparas i en variable 
% pos av storlek 50 x 150 x 1 x N
% där N är antalat urklippta toppar. Cirka 1400 stycken


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

%% Bilderna i pos är ganska brusiga. 
% Kanske är det bra att titta på lite utjämnade bilder
% Här skapas pos_smooth med sådana utsmetade bilder.  

if 1,
    pos_smooth = pos;
    for k = 1:size(pos_smooth,4);
        pos_smooth(:,:,:,k)=conv2(pos_smooth(:,:,:,k),ones(5,5)/25,'same');
    end
end
%% Här är ett montage av alla bilder i pos

figure(4);
subplot(1,1,1)
colormap(gray);
montage(pos);
