%% Using PCA to do 
% pos -> code -> pos_reconstruct
% pos -> code is encoding or dimensionality reduction
% code -> pos_reconstruct is decoding or unpacking of code

%% Do the PCA
KK = 20; % using 40 singular components
model = kalle_make_pca(pos,KK);

%% calculate dimensionality reduction for data

code = kalle_pca_encode(model,pos);

%% Reconstruct from code

pos_reconstruct = kalle_pca_decode(model,code);

%% Visualize standard deviation of each mode

figure(5);
plot(diag(model.s),'*');
xlabel('Mode nr');
ylabel('Standard deviation');

%% Visualize mean and mode images

figure(6);
subplot(4,4,1);
colormap(gray);
imagesc(reshape(model.data_mean,50,150));
title('Mean');
axis xy;
for k = 1:min(KK,15);
    subplot(4,4,k+1);
    colormap(gray);
    imagesc(reshape(model.u(:,k),50,150));
    str = ['Dimension ', num2str(k)];
    title(str);
    axis xy;
end


%% Plottar och f?rgkodar efter klass

figure(7);
hold on
for dot = 1:length(index)

switch index(2,dot) %letar upp klass i index
    case 0
        plot(code(dot,1), code(dot,2),'+g'); %Normalt v?rde
    case 1
        plot(code(dot,1),code(dot,2),'+b');
    case 2
        plot(code(dot,1), code(dot,2),'+y');
    case 3
        plot(code(dot,1), code(dot,2),'+r');

        
%text(code(dot,1),code(dot,2), string(index(3,dot)));
end
%text(code(:,1),code(:,2), string(index(3,:)));
end

xlabel('First dimension');
ylabel('Second dimension');
title('Using two dimensions');

%% Plottar i tre dimensioner

figure(8);

hold off
plot3(code(:,1), code(:,2), code(:,3),'+b'); %Normalt v?rde
 
title('Using three dimensions');

%% Visualize code for each image

figure(9);
plot(code(:,1),code(:,2),'*b');
text(code(:,1),code(:,2), string(index(3,:))); 
title('Using two dimensions');

figure(10);
hold off;
plot3(code(:,1),code(:,2), code(:,3), 'b*');
text(code(:,1),code(:,2), code(:,3), string(index(3,:)));    % plottar pulsatilt index
%text(code(:,1),code(:,2), code(:,3), string(index(1,:)));   %plottar fr?n vilken inspelning v?rdena kommer ifr?n
%text(code(:,1),code(:,2), code(:,3), string(index));
title('Using three dimensions');

%% Plotta l?gre dimensioner

figure(11);

hold on
for dot = 1:length(index)

switch index(2,dot) %letar upp klass i index
    case 0
        plot(code(dot,4),code(dot,5), '+g'); %Normalt v?rde
    case 1
        plot(code(dot,4),code(dot,5),'+b');
    case 2
        plot(code(dot,4),code(dot,5),'+y');
    case 3
        plot(code(dot,4),code(dot,5),'+r');

%text(code(:,1),code(:,2), string(index));
end
end

title('Using two dimensions');

%% Compare original with reconstruction

%for k = 1:size(pos,4);
for k = 50:50:500;
    figure(12);
    
    subplot(1,2,1);
    colormap(gray);
    imagesc(pos(:,:,:,k));
    title('Original');
    axis xy;
    subplot(1,2,2);
    colormap(gray);
    imagesc(pos_reconstruct(:,:,:,k));
    title(['Reprojection using ' num2str(KK) ' compponents.']);
    axis xy;
    pause(1);
end


%% Plottar och f?rgkodar efter outcome

figure(13);        
hold on
for dot = 1:length(index)

switch index(7,dot) %letar upp klass i index
    case 0
        plot(code(dot,1), code(dot,2),'+g'); %Normalt v?rde
    case 1
        plot(code(dot,1),code(dot,2),'+r');


        
%text(code(dot,1),code(dot,2), string(index(3,dot)));
end
%text(code(:,1),code(:,2), string(index(3,:)));
end

xlabel('First dimension');
ylabel('Second dimension');
title('Neonatal Intensive Care Unit');
