import numpy as np
import math
import json
import csv


class PhotonPropagation:
    def __init__(self, x_0, y_0, n_photons, arr_times):
        self.x_0 = x_0
        self.y_0 = y_0
        self.n_photons = n_photons
        self.arr_times = arr_times
        self.z_0 = 0
        self.params = self.read_params()

    def read_params(self):
        with open('pmt_simulation/simulation_params.json', 'r') as file:
            params = json.load(file)
        return params

    def sim_pmt_hits(self, x_0, y_0, n_photons):
        hits = {key: 0 for key in ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']}
        pmt_positions = self.params['pmt_positions']
        pmt_radius_sq = self.params['pmt_radius']**2

        for _ in range(n_photons):
            n = 3
            components = [np.random.normal() for i in range(n)]
            r = math.sqrt(sum(x*x for x in components))
            u = [x/r for x in components]
            if u[1] < 0:
                continue

            t = (self.params['dist_gem_pmt'] - self.z_0) / u[2]
            x = x_0 + t * u[0]
            y = y_0 + t * u[1]

            for pmt_name, pmt_pos in pmt_positions.items():
                dx = x - pmt_pos['x']
                dy = y - pmt_pos['y']
                if dx**2 + dy**2 < pmt_radius_sq:
                    hits[pmt_name] += 1
                    break

        return hits

    def sim_pmt_hits_with_map(self, x_0, y_0, n_photons):
        pmt_positions = self.params['pmt_positions']
        
        sigma_x = 130
        sigma_y = 130
        
        hits = {}

        for pmt_name, pmt_pos in pmt_positions.items():
            x_pmt, y_pmt = pmt_pos['x'], pmt_pos['y']

            prob = (
                0.00169 * np.exp(-0.5 * ((x_0 - x_pmt) / sigma_x) ** 2) *
                np.exp(-0.5 * ((y_0 - y_pmt) / sigma_y) ** 2)
            )

            hits[pmt_name] = np.random.poisson(prob * n_photons)

        return hits

    def sim_pmt_hits_with_eq(self, x_0, y_0, n_photons):
        pmt_positions = self.params['pmt_positions']
        
        hits = {}
        
        for pmt_name, pmt_pos in pmt_positions.items():
            x_pmt, y_pmt, z_pmt = pmt_pos['x'], pmt_pos['y'], self.params['dist_gem_pmt']
            r_pmt = self.params['pmt_radius']
            n = 3.8
            R = np.sqrt((x_pmt - x_0) ** 2 + (y_pmt - y_0) ** 2 + (z_pmt - self.z_0) ** 2)
            
            hits[pmt_name] = np.random.poisson(n_photons * (r_pmt ** 2) * (z_pmt ** 2) / (4 * (R ** n)))
            
        return hits

    def pmt_hits(self, mode):   
        #Mode = 0: Use equation R^n
        #Mode = 1: Use photon by photon propagation
        #Mode = 2: Use map
        
        hits = {}

        for i, (x, y, p, t) in enumerate(zip(self.x_0, self.y_0, self.n_photons, self.arr_times)):
            voxel_name = 'voxel_{}'.format(i)
            if   (mode==0):
                hits[voxel_name] = self.sim_pmt_hits_with_eq(x, y, p)
            elif (mode==1):
                hits[voxel_name] = self.sim_pmt_hits(x, y, p)
            elif (mode==2):
                hits[voxel_name] = self.sim_pmt_hits_with_map(x, y, p)
            hits[voxel_name]['arrival_time'] = t
        return hits
