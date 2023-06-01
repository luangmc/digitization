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
        with open('simulation_params.json', 'r') as file:
            params = json.load(file)
        return params

    def random_three_vector(self):
        """
        Generates a random 3D unit vector (direction) with a uniform spherical distribution
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        :return:
        """
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)
        theta = np.arccos(costheta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x ,y, z

    def randomvector(self, n):
        components = [np.random.normal() for i in range(n)]
        r = math.sqrt(sum(x*x for x in components))
        v = [x/r for x in components]
        return v

    def sim_pmt_hits(self, x_0, y_0, fotons):
    hits1, hits2, hits3, hits4, no_hits = 0, 0, 0, 0, 0

    for i in range(fotons):            
        if i%100==0:
            print("Propagate photon n. ",i)

        #u1, u2, u3 = self.random_three_vector()
        u1, u2, u3 = self.randomvector(3)

        if u2 < 0:
            continue

        t = (self.params['dist_gem_pmt'] - self.z0)/u3
        x = x_0 + t * u1
        y = y_0 + t * u2

        if ((x - self.params['pmt_positions']['pmt_1']['x'])**2 +  
            (y - self.params['pmt_positions']['pmt_1']['z'])**2) < self.params['pmt_radius']**2:
            hits1 += 1
        elif ((x - self.params['pmt_positions']['pmt_2']['x'])**2 +  
              (y - self.params['pmt_positions']['pmt_2']['z'])**2) < self.params['pmt_radius']**2:
            hits2 += 1
        elif ((x - self.params['pmt_positions']['pmt_3']['x'])**2 +  
              (y - self.params['pmt_positions']['pmt_3']['z'])**2) < self.params['pmt_radius']**2:
            hits3 += 1
        elif ((x - self.params['pmt_positions']['pmt_4']['x'])**2 +  
              (y - self.params['pmt_positions']['pmt_4']['z'])**2) < self.params['pmt_radius']**2:
            hits4 += 1
        else: 
            no_hits += 1

    return {'pmt1': hits1, 'pmt2': hits2, 'pmt3': hits3, 'pmt4': hits4}


    def sim_pmt_hits_with_map(self, x_0, y_0, photons):
        pmt_parameters = {
            'pmt1': {'mean_x': 312, 'mean_y': 312},
            'pmt2': {'mean_x': 312, 'mean_y': 42},
            'pmt3': {'mean_x': 42, 'mean_y': 42},
            'pmt4': {'mean_x': 42, 'mean_y': 312}
        }
        sigma_x = 130
        sigma_y = 130
        
        pmt_hits = {}
        
        for pmt, params in pmt_parameters.items():
            print("pmt n.", pmt)
            mean_x = params['mean_x']
            mean_y = params['mean_y']
            
            prob = (
                0.00169 * np.exp(-0.5 * ((x_0 - mean_x) / sigma_x) ** 2) *
                np.exp(-0.5 * ((y_0 - mean_y) / sigma_y) ** 2)
            )
            
            pmt_hits[pmt] = np.random.poisson(prob * photons)
        
        return pmt_hits

    def pmt_hits(self):
        hits = {}
        for i in range(len(self.x0)):            
            if i%50==0:
                print("Voxel n. ", i)
            hits['cluster_{}'.format(i)] = self.sim_pmt_hits_with_map(self.x0[i], 
                                                              self.y0[i], 
                                                              self.n_fotons[i])
            hits['cluster_{}'.format(i)]['arrival_time'] = self.arr_times[i]
                                                            
        return hits
