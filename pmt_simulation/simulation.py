import numpy as np
import json
from scipy.stats import exponnorm

class SignalSimulation:
    def __init__(self, hits_dict):
        self.ptc_hits = hits_dict
        self.t = np.array([])
        self.params = self.read_params()
        self.set_t()
        self.noise_covM = np.load(self.params['noise_path'])
        self.win_noise()

    def read_params(self):
        with open('pmt_simulation/simulation_params.json', 'r') as file:
            params = json.load(file)
        return params

    def quantum_efficiency(self, nr_photons):
        probs = [1 - self.params['quantum_efficiency'],
                 self.params['quantum_efficiency']]
        effc = np.random.choice(2, nr_photons, p=probs)
        self.nr_hits = effc.sum()

    def set_t(self):
        self.t = np.linspace(-365, 1000, self.params['window_len'])

    def gaussian(self, x, amp, mean, std):
        '''Function to generate a gaussian'''
        return amp * np.exp( - (1/2)*((x-mean)/std)**2)

    def fwhm2std(self, fwhm):
        '''Function to converts FWHM to standard deviation'''
        return fwhm/(2*np.sqrt(2*np.log(2)))

    def transit_time(self):
        mu = self.params['pmt_time_response']['transit_time']
        fwhm = self.params['pmt_time_response']['transit_time_spread']
        sigma = self.fwhm2std(fwhm)
        return np.random.normal(mu, sigma, 1)

    def signal_time(self):
        arr_time = [self.ptc_hits[t_voxel]['arrival_time']
                     for t_voxel in self.ptc_hits.keys()]
        return np.cumsum(arr_time)

    def voxel_signal(self, arr_time):
        amp = self.params['pmt_signal']['amplitude']\
              + (np.random.uniform(-1,1) *
                 self.params['pmt_signal']['amplitude_dispersion'] *
                 self.params['pmt_signal']['amplitude'])
        std = self.params['pmt_signal']['sigma']
        sig = np.array([])
        mean = self.transit_time()[0] + arr_time
        sig = self.gaussian(self.t, amp, mean, std)
        return sig

    def gen_signal(self, nr_photons, arr_time):
        cl_sig = np.array([self.voxel_signal(arr_time) for _ in range(nr_photons)])
        self.recoil_signal = np.sum(cl_sig, axis=0)

    def noise_gen(self, CovarianceMatrix, size):
        covM = CovarianceMatrix
        mean = np.zeros(np.shape(covM)[0])
        noise = np.random.multivariate_normal(mean, covM, size)
        return noise

    def win_noise(self):
        self.noise = self.noise_gen(self.noise_covM, 4)

    def pmt_signal(self, nr_photons, arr_time):
        #self.quantum_efficiency(nr_photons)
        #self.gen_signal(self.nr_hits, arr_time)
        # The quantum efficiency is now applied together with glass transmission before simulating the pmt
        self.gen_signal(self.nr_photons, arr_time)
        return self.recoil_signal

    def simulated_signals(self):
        pmts = ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']
        signal = {key: np.zeros([self.params['window_len']]) for key in pmts}
        arrival_time = self.signal_time()
        voxel_keys = list(self.ptc_hits.keys())
        for pmt in pmts:
            signal_aux = np.array([self.pmt_signal(self.ptc_hits[voxel][pmt],
                                                   arrival_time[i])
                                   for i, voxel in enumerate(voxel_keys)],
                                  dtype=object)
            signal[pmt] = np.sum(signal_aux, axis=0) + \
                          self.noise[pmts.index(pmt)]
        signal['time'] = self.t
        return signal
