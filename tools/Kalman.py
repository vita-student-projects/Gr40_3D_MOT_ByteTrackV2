import numpy as np

class KalmanFilter(object):
    def __init__(self, state, dt, tracking_name, Covariance):
        self.dt = dt
        self.E = state      # position as (x, y, z, l, w, h, θ, vx, vy, vz) 
        self.A = np.array([[1, 0, 0, 0, 0, 0, 0, dt, 0, 0], # The position in x depends on the speed in x with respect to t
                           [0, 1, 0, 0, 0, 0, 0, 0, dt, 0], # The position in y depends on the speed in y with respect to t
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, dt], # The positino in z depends on the speed in z with respect to t
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ]])
        
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], #### a voir sil faudrait pas ajouter un calcul d'angle ici
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ]]) # Observation matrix. The speed is not in it, hence the 7x10 matrix, as it is not something we are trying to estimate.
        self.Q = Covariance.Q[tracking_name]  # a compute avec la fonction de lautre paper
        self.R = Covariance.R[tracking_name]  # a compute avec la fonction de lautre paper
        self.P = Covariance.P[tracking_name]  # a compute avec la fonction de lautre paper
        self.i = None


    def predict(self): 
        #####################################################
        # Updates the state estimation vector aswell as the state covariance vector
        #####################################################
        self.E = np.dot(self.A, self.E) 
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.E


    def update(self, state, det_score=None):
        #####################################################
        # 
        #####################################################
        y = state[:-3] 
        self.i = y - self.E[:-3] 
        
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R 
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))      
        
        self.E = self.E + np.dot(K, self.i) 
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        if det_score is not None:
            self.R = 100*(1-det_score)**2 * self.R