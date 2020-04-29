%% Do the PCA
KK = 10; % using 40 singular components
model = kalle_make_pca(pos,KK);

%% calculate dimensionality reduction for data

code = kalle_pca_encode(model,pos);

%% Compare Points with same PI in one lower dimension.
figure(1);
upperPI = 1.7;
lowerPI = 1.65;

for column = 1:10
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

%% Compare points with sama PI in two lower dimensions

%% Compare Points with same PI in one lower dimension.
figure(2);
upperPI = 1.7;
lowerPI = 1.65;

for column = 1:10
    % subplot(3,3,i);
    hold on
    
    for dot = 1:length(index)
            
            if (lowerPI < index(3,dot) && index(3,dot) < upperPI)
                plot(code(dot,column),code(dot,column + 1),'r*');
                text(code(dot,column),code(dot,column + 1), string(index(3,dot)));
            end
    end
    plot(code(:,coolumn),0.01,'b*');
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