%% L?s in ljudfilsnamn fr?n en katalog

%Kalle
%datafolder = '/Users/kalle/Documents/projekt/filkand_2020_fosterdiagnostik/dataset1';
%datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset1';
%datafolder = '/Users/kalle/Documents/projekt/kand_2020_fosterdiagnostik/dataset2';

%Johan
datafolder = 'C:\Users\Johan\OneDrive\Skrivbord\Dataset_ljudfiler';

%Henrik
%datafolder = '/Users/97hen/Desktop/Dataset_ljudfiler';
%sheet = '/Users/97hen/Desktop/matlab_sheet';
sheet = 'C:\Users\Johan\OneDrive\Skrivbord\Kandidat\Klasser';

a = dir(fullfile(datafolder,'*.wav'));
sheetNums = xlsread(sheet, 'B:B');    %l?ser in klasserna fr?n excellfil

%% L?s in varje ljudfil och g?r inledande analys
% bl a r?kna ut vilken kanal som ?r dominant. 

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
        facit(i).kanal = sorti;  % H?r ?r f?rsta talet den dominanta kanalen
        
        
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
        x = x.*(((1:50)')*ones(1,size(x,2))); % vikta mer linj?rt fr?n toppen
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
        
        facit(i).lokmax = lokmax; % ?r ?r index f?r topparna
        
        %         if facit(i).class == 1,
        %             facit(i).lokmax = lokmax;
        %         else
        %             facit(i).lokmax = [];
        %         end
        %pause(0.1);
        
    end;
    
end;

%% Liten test f?r att se om avst?ndet mellan tv? toppar ?r ungef?r samma
% Anv?nd medianen av avst?nden som skattning av storleken.

for i = 1:length(a);
    %     figure(1);
    %     clf;
    %     plot(diff(facit(i).lokmax),'*');
    %     axis([0 20 0 2000]);
    %     pause
    facit(i).period = median(diff(facit(i).lokmax));
end;

%% H?r g?rs urklippen. Jag skapara bilder av storlek 50 x 150 
% f?r varje topp. Dessa sparas i en variable 
% pos av storlek 50 x 150 x 1 x N
% d?r N ?r antalet urklippta toppar. Cirka 1400 stycken


pos = zeros(50,150,1,0);
%neg = zeros(32,32,1,0);
index = [];


for i = 1:length(a);
    
    if (i ~= 40)
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
            % sista tar inte h?nsyn till att sista delen av signalen ?r tyst
            if (mid>(w2+10)) & (mid< (length(x)-w2-10)), 
                cutout = x(1:50,(mid-w2):(mid+w2-1));
                cutout = conv2(cutout,ones(1,20)/20,'same');
                cutout = cutout(:,10:20:end);
                cutout = cutout/max(cutout(:));
                cutout = imresize(cutout,[50 150]);
                figure(3); clf;
                imagesc(cutout);
                axis xy
                title([num2str(i) ' - ' num2str(mid)]);
                % pause(0.1);
                
                
                pos = cat(4,pos,cutout);
                p = size(index,2) +1;
                index(1,p) = i;
                index(2,p) = sheetNums(i);
            end

            end
        end
    end
end


%% Bilderna i pos ?r ganska brusiga. imagesc
% Kanske ?r det bra att titta p? lite utj?mnade bilder
% H?r skapas pos_smooth med s?dana utsmetade bilder.  

if 1,
    pos_smooth = pos;
    for k = 1:size(pos_smooth,4);
        pos_smooth(:,:,:,k)=conv2(pos_smooth(:,:,:,k),ones(5,5)/25,'same');
    end
end
%% H?r ?r ett montage av alla bilder i pos

figure(4);
subplot(1,1,1)
colormap(gray);
montage(pos);
