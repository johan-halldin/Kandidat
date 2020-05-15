%% J?mf?r separation mellan olika f?rdigtr?nade encoders
%ladda f?rst in en tr?nad encoder och dataPrepp

code = predict(encoder,pos);
pos_reconstruct = predict(decoder,reshape(code',1,1,KK,size(code,1)));

% de ?r inte l?ngre sorterade i "viktigast" ordning
figure(1);
firstDim = 1;
secondDim = 4;
hold on
for dot = 1:length(index)
    
    switch index(5,dot) %letar upp klass i index
        case 0
            plot(code(dot,firstDim), code(dot,secondDim),'+g'); %Normalt v?rde
        case 1
            plot(code(dot,firstDim),code(dot,secondDim),'+b');
        case 2
            plot(code(dot,firstDim), code(dot,secondDim),'+y');
        case 3
            plot(code(dot,firstDim), code(dot,secondDim),'+r');
    end
end

% plot(code(:,9),code(:,19),'b*');
% text(double(code(:,9)),double(code(:,19)), string(index(2,:)));
title('Using three dimensions');

%% Compare original with reconstruction

%for k = 1:size(pos,4);
for k = 50:50:1000;
    figure(2);
    subplot(1,2,1);
    colormap(gray);
    imagesc(pos(:,:,:,k));
    axis xy
    title('Original');
    subplot(1,2,2);
    colormap(gray);
    imagesc(pos_reconstruct(:,:,:,k));
    title(['Reprojection using ' num2str(KK) ' compponents.']);
    axis xy
    pause(0.2);
end

%%
mm = mean(code);
ss = std(code);
ee = eye(KK);
figure(3);
for k = 1:3 %KK
    subplot(1,2,1);
    colormap(gray);
    meanimage = predict(decoder,reshape(mm,[1 1 KK]));
    imagesc(meanimage);
    axis xy
    for th = linspace(0,2*pi,100);
        subplot(1,2,2);
        colormap(gray);
        meanimage = predict(decoder,reshape(mm+cos(th)*ss.*ee(k,:),[1 1 KK]));
        imagesc(meanimage);
        title(['Mode: ' num2str(k) ' of ' num2str(KK)]);
        axis xy
        pause(0.0003);
    end;
    %pause;
end;

%% Compare Points with same PI
figure(4);
upperPI = 1.4;
lowerPI = 1;

for column = 1:3 %KK
    hold on
    for dot = 1:length(index)
        
        if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
            switch index(5,dot)
                case 1
                    plot(code(dot,column),0,'r*');
                    
                case 0
                    plot(code(dot,column),0,'g*');
            end
            
        end
    end
    plot(code(:,column),0.01,'b*');
    title(string(column));
    
    pause;
    clf
end

%%
%% Compare Points with same PI in one lower dimension. OUTCOME
figure(14);
upperPI = 1.4;
lowerPI = 1;
column1 = 1;
column2 = 2;

for column = 1:3    
    subplot(1,2,2)
    hold on
    for dot = 1:length(index)
        
        if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
            switch index(5,dot)
                case 1
                    plot(code(dot,column),0,'r+');
                    
                case 0
                    plot(code(dot,column),0,'g+');
            end
            set(gca,'ytick',[])
            
        end
    end
    plot(code(:,column),0.01,'b+');
    set(gca,'ytick',[])
    xlabel('this dimension');
    title(string(column));
    
    subplot(1,2,1)
    hold on
    for dot = 1:length(index)
        
        if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
            plot(code(dot,1),code(dot,2),'b+');
        else
             plot(code(dot,1),code(dot,2),'y+');
        end
       
    end
    xlabel('First dimension');
    ylabel('Second dimension');
    suptitle('Recordings from group of similar PI colorcoded by outcome in X dimension');
    pause;
    clf
end
