import numpy as np
import math
import json
import csv


class PhotonPropagation:
    def __init__(self, x0, y0, n_fotons, arr_times):
        self.x0 = x0
        self.y0 = y0
        self.n_fotons = n_fotons
        self.arr_times = arr_times
        self.z0 = 0
        self.params = self.read_params()

    def read_params(self):
        with open('pmt_simulation/simulation_params.json', 'r') as file:
            params = json.load(file)
        return params

    def sim_pmt_hits(self, x_0, y_0, fotons):
            hits = {key: 0 for key in ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']}
            pmt_positions = self.params['pmt_positions']
            pmt_radius_sq = self.params['pmt_radius']**2

            for _ in range(fotons):
                n = 3
                components = [np.random.normal() for i in range(n)]
                r = math.sqrt(sum(x*x for x in components))
                u = [x/r for x in components]
                if u[1] < 0:
                    continue

                t = (self.params['dist_gem_pmt'] - self.z0) / u[2]
                x = x_0 + t * u[0]
                y = y_0 + t * u[1]

                for pmt_name, pmt_pos in pmt_positions.items():
                    dx = x - pmt_pos['x']
                    dy = y - pmt_pos['y']
                    if dx**2 + dy**2 < pmt_radius_sq:
                        hits[pmt_name] += 1
                        break

            return hits

    def sim_pmt_hits_with_map(self, x_0, y_0, photons):
        pmt_pos = self.params['pmt_positions']
        sigma_x = 130
        sigma_y = 130

        pmt_hits = {}

        for pmt, params in pmt_pos.items():
            mean_x = params['x']
            mean_y = params['y']

            prob = (
                0.00169 * np.exp(-0.5 * ((x_0 - mean_x) / sigma_x) ** 2) *
                np.exp(-0.5 * ((y_0 - mean_y) / sigma_y) ** 2)
            )

            pmt_hits[pmt] = np.random.poisson(prob * photons)

        return pmt_hits

    def sim_pmt_hits_with_eq(self, pmt_position, x0, y0, n_fotons):
        x_pmt, y_pmt, z_pmt = pmt_position['x'], pmt_position['y'], self.params['dist_gem_pmt']
        r_pmt = self.params['pmt_radius']
        n=3.8
        R = np.sqrt((x_pmt - x0) ** 2 + (y_pmt - y0) ** 2 + (z_pmt - self.z0) ** 2)
        return np.random.poisson(n_fotons * (r_pmt ** 2) * (z_pmt ** 2) / (4 * (R ** n)), 1)

    def pmt_hits(self, mode):
        #Mode = 0: Use equation R^n
        #Mode = 1: Use photon by photon propagation
        #Mode = 2: Use map
        hits = {}
        pmts_list = ['pmt_1', 'pmt_2', 'pmt_3', 'pmt_4']

        for i, (x, y, f, t) in enumerate(zip(self.x0, self.y0, self.n_fotons, self.arr_times)):
            voxel_name = 'voxel_{}'.format(i)
            if (mode==0):
                hits[voxel_name] = {key: 0 for key in pmts_list}
                for p in pmts_list:
                  pmt_position = self.params["pmt_positions"][p]
                  hits[voxel_name][p] = int(self.sim_pmt_hits_with_eq(pmt_position, x, y, f))
            elif (mode==1):
                hits[voxel_name] = self.sim_pmt_hits(x, y, f)
            elif (mode==2):
                hits[voxel_name] = self.sim_pmt_hits_with_map(x, y, f)
            hits[voxel_name]['arrival_time'] = t
        return hits
