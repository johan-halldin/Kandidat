%% Do the PCA
KK = 10; % using 40 singular components
model = kalle_make_pca(pos,KK);

%% calculate dimensionality reduction for data

code = kalle_pca_encode(model,pos);

%%
figure(1);
firstCol = 3;
secondCol = 4;
upperPI = 1.7;
lowerPI = 1.68;



for i = 1:3
    for k = 1:3
        firstCol = i;
        secondCol = k;
        subplot(5,1,(i+k)-1);
        hold on
        plot(code(:,firstCol),code(:,secondCol), 'b*');
        for dot = 1:length(index)
            
            if lowerPI < index(3,dot) && index(3,dot) < upperPI
                text(code(dot,firstCol),code(dot,secondCol), string(index(3,dot)));
                hold off
            end
        end
    end
end