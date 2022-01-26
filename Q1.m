% MATLAB 2020b
%signal specification
 fo = 77e9;
 alpha = 60e12;
 c = 2.3e8;
 dt = 5.33e-8;
 N= 7; %we assume that the number of target is seven
 to = 5.7e-6;
  
% Building the model
learningRate = 0.0000001;
repetition = 1000;
 
% Storing the values in  matrices  
%filename = '\\depot.engr.oregonstate.edu\users\alsulaiz\Windows.Documents\Desktop\signal.csv';
filename = 'C:/Users/Bobby/Desktop/signal.csv';
M = readtable(filename);
M = M{:,:};
M(isnan(M))=0;
M(:,2) = M(:,2)*1i; %change the second column to imagenary
S = [sum(M(:,[1 2]),2)]; % sum img and real
S(isnan(S)) = 0;
n = length(S);
S_pred = zeros(n,1) ;

% Initialize the Rn vector
rng('default')
a = 1;
b = 10;
R = (b-a).*rand(N,1) + a;
An = rand(N,1) + rand(N,1)*1i;
  

% Creating a matrix of zeros for storing our cost function history
costHistory = zeros(n, 1);


for counter = 1:repetition
 for m = 1:n       
    t = to + m*dt; %find t value
    S_pred(m) =  sum (An.*(exp(R.*1i*4*pi*(fo + alpha*t)/c)))  ;  % The current predicted value of S
 end
    divr = (1i*4*pi*(fo + alpha*t)/c)*sum(An.*exp(R.*1i*4*pi*(fo + alpha*t)/c))   ; % Partial derivative  wrt r
    
    divAn = sum(exp(R.*1i*4*pi*(fo + alpha*t)/c))   ; % Partial derivative  wrt An
    
    D_r =   (-2/n) * sum(abs(divr)*(abs(S) - abs(S_pred))) ;  %Gradient size for R (real)
    
    D_r_An =   (-2/n) * sum(divAn * (S - S_pred)) ; % Gradient size for An (complix)
    
    R = R - learningRate * D_r;  % Update R
    
    An = An -  learningRate * D_r_An;  % Update An
    
    costHistory(counter) = sum(( abs(S_pred)  - abs(S)).^2) / (2 * n);            
  
end
