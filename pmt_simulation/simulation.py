import numpy as np
import json
import scipy
import multiprocessing
import concurrent

class SignalSimulation:
    def __init__(self, hits_dict):
        self.ptc_hits = hits_dict
        self.params = self.read_params()
        self.digitizers = self.params['digitizers']
        self.nJobs = self.params['nJobs']
        self.fast_window_len = self.params['fast_window_len']
        self.slow_window_len = self.params['slow_window_len']
        self.Fs_fast = self.params['fast_freq']
        self.Fs_slow = self.params['slow_freq']
        self.set_t()
        self.gen_noise()
        
    def read_params(self):
        with open('pmt_simulation/simulation_params.json', 'r') as file:
            params = json.load(file)
        return params
    
    def signal_time(self):
        arr_time = [self.ptc_hits[t_voxel]['arrival_time']
                     for t_voxel in self.ptc_hits.keys()]
        return np.cumsum(arr_time)
    
    def set_t(self):
        # Fast Digitizer
        self.sample_fast = np.arange(self.fast_window_len)
        self.t_fast = self.sample_fast / self.Fs_fast * 1e9 # Converted to ns
        
        # Slow Digitizer
        self.sample_slow = np.arange(self.slow_window_len)
        self.t_slow = self.sample_slow / self.Fs_slow * 1e9 # Converted to ns
    
    def quantum_efficiency(self, nr_photons):
        probs = [1 - self.params['quantum_efficiency'],
                 self.params['quantum_efficiency']]
        effc = np.random.choice(2, nr_photons, p=probs)
        self.nr_hits = effc.sum()
                    
    def compute_noise(self, psd, digitizer):
        if digitizer == 'Fast':
            fs = self.Fs_fast
            N = self.fast_window_len
        if digitizer == 'Slow':
            fs = self.Fs_slow
            N = self.slow_window_len

        PSD_pos= psd
        PSD_neg = PSD_pos[-2:0:-1]
        PSD = np.concatenate((PSD_pos, PSD_neg))
        phase = np.random.uniform(-np.pi, np.pi, len(PSD_pos))
        Nf_pos = np.sqrt(PSD_pos * fs / N) * np.exp(1j * phase)
        Nf_neg = np.conj(Nf_pos[-2:0:-1])
        Nf = np.concatenate((Nf_pos, Nf_neg))
        return np.fft.ifft(Nf).real * N

    def gen_noise(self):
        pmts = ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']
        self.fast_noise = {key: np.zeros([self.fast_window_len]) for key in pmts}
        self.slow_noise = {key: np.zeros([self.slow_window_len]) for key in pmts}
        for pmt in pmts:
            fast_psd = np.load(self.params['fast_noise_path'][pmt])
            slow_psd = np.load(self.params['slow_noise_path'][pmt])
            self.fast_noise[pmt] = self.compute_noise(fast_psd, 'Fast')
            self.slow_noise[pmt] = self.compute_noise(slow_psd, 'Slow')  
        
    def fwhm2std(self, fwhm):
        '''Function to converts FWHM to standard deviation'''
        return fwhm/(2*np.sqrt(2*np.log(2)))

    def transit_time(self):
        mu = self.params['pmt_time_response']['transit_time']
        fwhm = self.params['pmt_time_response']['transit_time_spread']
        sigma = self.fwhm2std(fwhm)
        return np.random.normal(mu, sigma, 1)
    
    def expgaussian(self, x, amp, cen, sig, lamb):
        '''Function to generate a Exp. modified gaussian'''
        pdf = lamb / 2 * np.exp(lamb * (cen - x + (lamb * sig**2 / 2))) * scipy.special.erfc((cen + (lamb * sig**2) - x)/(sig * np.sqrt(2)))
        wf = amp * (pdf/max(pdf))
        return wf
    
    def spe_signal(self, arr_time, fast_wf_aux, slow_wf_aux):
        amp = np.random.normal(self.params['pmt_signal']['amplitude'], self.params['pmt_signal']['amplitude_dispersion'])
        std = self.params['pmt_signal']['sigma']
        lamb = self.params['pmt_signal']['lambda']
        mean = self.transit_time()[0] + arr_time
        
        if self.digitizers in ['Both', 'Fast']:
            fast_wf_aux += self.expgaussian(self.t_fast, amp, mean, std, lamb)
        if self.digitizers in ['Both', 'Slow']:
            slow_wf_aux += self.expgaussian(self.t_slow, amp, mean, std, lamb)

    def gen_signal(self, nr_photons, arr_time, fast_wf_aux, slow_wf_aux):
        #self.quantum_efficiency(nr_photons)
        # Quantum efficiency already applied before the simulation
        for i in range(nr_photons):
            self.spe_signal(arr_time, fast_wf_aux, slow_wf_aux)        
    
    def quantization(self, signal):
        n_bits = 12
        num_levels = 2 ** n_bits
        
        # Signal is generated in Volts, convert to ADC levels
        signal = np.round(signal * num_levels)
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        quantization_interval = (signal_max - signal_min) / (num_levels - 1)

        signal_scaled = (signal - signal_min) / quantization_interval
        signal_rounded = np.round(signal_scaled)
        signal_rounded = np.clip(signal_rounded, 0, num_levels - 1)

        # Convert back to Volts
        quantized_signal = (signal_rounded * quantization_interval + signal_min) / (num_levels - 1)
    
        return quantized_signal

    def pmt_signal(self, pmt, voxel_keys, arrival_time, fast_signal, slow_signal):
        fast_wf_aux = np.zeros(self.fast_window_len)
        slow_wf_aux = np.zeros(self.slow_window_len)
        for i, voxel in enumerate(voxel_keys):
            self.gen_signal(self.ptc_hits[voxel][pmt], arrival_time[i], fast_wf_aux, slow_wf_aux)
            
        # Fast waveform
        shift = 200
        fast_signal[pmt] = np.roll(fast_wf_aux, shift) + self.fast_noise[pmt]
        fast_signal[pmt] = self.quantization(fast_signal[pmt])

        # Slow waveform
        shift = 1500
        slow_signal[pmt] = np.roll(slow_wf_aux, shift) + self.slow_noise[pmt]
        slow_signal[pmt] = self.quantization(slow_signal[pmt])

    def simulated_signals(self):
        pmts = ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']
        arrival_time = self.signal_time()
        voxel_keys = list(self.ptc_hits.keys())     
        
        if self.nJobs == -1 or self.nJobs > len(pmts): # -1 uses all the cores available
            cpu_count = multiprocessing.cpu_count()
            self.nJobs = len(pmts) if cpu_count > len(pmts) else cpu_count
            
        if self.nJobs>1:
            print("Parallelizing in %d processes"%self.nJobs)
            manager = multiprocessing.Manager()
            fast_signal = manager.dict({key: np.zeros(self.fast_window_len) for key in pmts})
            slow_signal = manager.dict({key: np.zeros(self.slow_window_len) for key in pmts})        
            with concurrent.futures.ProcessPoolExecutor(self.nJobs) as executor:
                for pmt in pmts:
                    executor.submit(self.pmt_signal, pmt, voxel_keys, arrival_time, fast_signal, slow_signal)            
        else:
            fast_signal = {key: np.zeros([self.fast_window_len]) for key in pmts}
            slow_signal = {key: np.zeros([self.slow_window_len]) for key in pmts}
            for pmt in pmts:
                self.pmt_signal(pmt, voxel_keys, arrival_time, fast_signal, slow_signal)

        fast_signal['time'] = self.t_fast
        slow_signal['time'] = self.t_slow
        
        return fast_signal, slow_signal