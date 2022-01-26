% MATLAB 2020b
% Building the model
learningRate = 0.001;
repetition = 1500;

 
% Storing the values in  matrices
x = [10; 0; 1 ; 30; -4.5];
xm = mean(x);  %initial value of x

y = [ 0 ; 20 ; -1 ; -30 ; 5.5];
ym = mean(x);  %initial value of y

z = [ 0; 0; -20 ; 30 ; 6];
zm = mean(x);  %initial value of z

d = [18.37; 18.22; 31.47; 53.08; 4.35];

% Getting the length of our dataset
n = length(d);

% Creating a matrix of zeros for storing our cost function history
costHistory = zeros(repetition, 1);
 

% Running gradient descent

for i = 1:repetition        
    d_pred = sqrt((x-xm).^2 + (y-ym).^2 + (z-zm).^2) ;  % The current predicted value of Y
    
    divx = - (x-xm)/sqrt((x-xm).^2 + (y-ym).^2 + (z-zm).^2); % partial derivative  wrt x
    divy = - (y-ym)/sqrt((x-xm).^2 + (y-ym).^2 + (z-zm).^2); % partial derivative  wrt y
    divz = - (z-zm)/sqrt((x-xm).^2 + (y-ym).^2 + (z-zm).^2); % partial derivative  wrt z
    
    D_x = (-2/n) * sum(divx * (d - d_pred));  
    D_y = (-2/n) * sum(divy *(d - d_pred)) ;  
    D_z = (-2/n) * sum(divz *(d - d_pred)) ;  
    
    % Updating the parameters
    xm = xm - learningRate * D_x;  % Update x
    ym = ym - learningRate * D_y;  % Update y
    zm = zm - learningRate * D_z;  % Update z
   % costHistory(i) = sum((d_pred - d).^2) / (2 * n);         
    costHistory(i) = (d_pred - d)' * (d_pred - d)  / (2 * n); 
 end 
 

% Plotting the cost function

figure;
plot(costHistory, 1:repetition);
xlabel('cost');
ylabel('iteration');
title('Find the Position of the Radar'); 

 
 

