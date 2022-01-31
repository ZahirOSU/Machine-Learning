% Radar Specifications 
% Frequency of operation = 77GHz
% Alpha  = slop  = 60e12 Hz/s 
 
fc = 77e9;                          % Hz (carrier frequency) Operating carrier frequency of Radar 
slope = 60e12    ;                  % Hz/s             
c = 3e8;                            % m/s (speed of light)
max_range = 200;                    % m
range_resolution = 1;               % m
str_factor = 5.5;                       % sweep-to-roundtrip factor as a typical value for an FMCW radar system.

Bsweep = c/(2*range_resolution);         % Hz, bandwidth
Tchirp = str_factor * 2 * max_range/c ;  % s, chirp time 
                                 
%The number of chirps in one sequence. Its ideal to have 2^ value for he ease of running the FFT for Doppler Estimation. 
Nd = 4;                            

%The number of samples on each chirp. 
Nr = 256;                           

% Timestamp for running the displacement scenario for every sample on each chirp
t = linspace(0, Nd*Tchirp, Nr*Nd);  % total time for samples

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
Rx = S;

%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx  = zeros(1, length(t));          % transmitted signal
Mix = zeros(1, length(t));          % beat signal

 


% Running the radar scenario over the time. 

for d=1:length(t)         
    
    % For each time step we need update the transmitted signal. 
    t_tx = t(d);
    tx_phase = 2*pi*( fc*t_tx + 0.5*slope*t_tx^2 );   % unit-less
    
    
    Tx(d) = cos(tx_phase);
    
    
    % Now by mixing the Transmit and Receive generate the beat signal
    % This is done by element wise matrix multiplication of Transmit and
    % Receiver Signal
    Mix(d) = Tx(d) .* Rx(d);
    
end

% Reshape the vector into Nr*Nd array. Nr and Nd here would also define
% the size of Range and Doppler FFT respectively.
Mix = reshape(Mix, [Nr, Nd]);

% Run the FFT on the beat signal along the range bins dimension (Nr)and normalize.
sig_fft = fft(Mix, Nr) ./ length(Mix);

% Take the absolute value of FFT output
sig_fft = abs(sig_fft);

% Output of FFT is double sided signal, but we are interested in only
% one side of the spectrum. Hence we throw out half of the samples.
sig_fft = sig_fft( 1:(Nr/2) );

%[val, idx] = max(sig_fft);
%estimated_distance_to_target = idx;


% Plotting the range
figure('Name', 'Range from First FFT')

% Plot FFT output 
plot(sig_fft); 
axis([0, 140, 0, 1]);
ylim([0, 300])
grid minor;
xlabel('measured range [m]');
ylabel('amplitude');  
%% RANGE DOPPLER RESPONSE
% This will run a 2DFFT on the mixed signal (beat signal) output and
% generate a range doppler map.  


% Range Doppler Map (RDM) Generation.

Mix = reshape(Mix, [Nr, Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix, Nr, Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift(sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM);
RDM2_signal = zeros(size(RDM));
%%  implementation


for r = 1 : 128
    for d = 1 : 4
       % If the signal in the cell under test exceeds the
        % threshold, we mark the cell as hot by setting it to 1.
        % We don't need to set it to zero, since the array
        % is already zeroed out.
        
        threshold = 50;
        if (RDM(r, d) >= threshold)
            RDM2_signal(r, d) = 1; % ... or set to 1
        end
        
    end
end
figure
im = imagesc(RDM2_signal);
 