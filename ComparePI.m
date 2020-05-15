%% Do the PCA
KK = 20; % using 40 singular components
model = kalle_make_pca(pos,KK);

%% calculate dimensionality reduction for data

code = kalle_pca_encode(model,pos);

%% Compare Points with same PI in one lower dimension.
figure(1);
upperPI = 1.2;
lowerPI = 0.9;

for column = 1:3
    % subplot(3,3,i);
    hold on
    
    for dot = 1:length(index)
        
        if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
            plot(code(dot,column),0,'r*');
            text(code(dot,column),0, string(index(3,dot)));
        end
    end
    plot(code(:,column),0.01,'b*');
    title(string(column));
    pause;
    clf
end

%% Compare Points with same PI in one lower dimension.
figure(2);
upperPI = 1.2;
lowerPI = 0.9;

for column = 1:3
    % subplot(3,3,i);
    hold on
    
    for dot = 1:length(index)
        
        if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
            plot(code(dot,column),code(dot,column + 1),'r*');
            text(code(dot,column),code(dot,column + 1), string(index(3,dot)));
        end
    end
    plot(code(:,column),0.01,'b*');
    title(string(column));
    pause;
    clf
end

%%
figure(3);
firstCol = 3;
secondCol = 4;

for i = 1:3
    for k = 1:3
        firstCol = i;
        secondCol = k;
        subplot(5,1,(i+k)-1);
        hold on
        plot(code(:,firstCol),code(:,secondCol), 'b*');
        for dot = 1:length(index)
            
            if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
                text(code(dot,firstCol),code(dot,secondCol), string(index(3,dot)));
                hold off
            end
        end
        
    end
end


%% Compare Points with same PI in one lower dimension. OUTCOME
figure(14);
upperPI = 1.4;
lowerPI = 1;

for column = 1:20            
    subplot(1,2,1)
    hold on
    for dot = 1:length(index)
        
        if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
            switch index(5,dot)
                case 1
                    plot(code(dot,column),0,'r*');
                    text(code(dot,column),0.0005,string(index(3,dot)));
                    
                case 0
                    plot(code(dot,column),0,'g*');
            end
            
        end
    end
    plot(code(:,column),0.01,'b*');
    title(string(column));
    
    subplot(1,2,2)
    hold on
    for dot = 1:length(index)
        
        if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
            plot(code(dot,1),code(dot,2),'b*');
        else
             plot(code(dot,1),code(dot,2),'y*');
        end
       
    end
    
    pause;
    clf
end

