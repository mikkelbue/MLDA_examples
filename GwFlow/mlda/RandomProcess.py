import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial import distance_matrix


class SquaredExponential:
    def __init__(self, coords, mkl, lamb):
        """
        This class sets up a random process
        on a grid and generates
        a realisation of the process, given
        parameters or a random vector.
        """
        
        # Internalise the grid and set number of vertices.
        self.coords = coords
        self.n_points = self.coords.shape[0]
        self.eigenvalues = None
        self.eigenvectors = None
        self.parameters = None
        self.random_field = None
        
        # Set some random field parameters.
        self.mkl = mkl
        self.lamb = lamb
        
        self.assemble_covariance_matrix()
        
    def assemble_covariance_matrix(self):
        
        # Create a snazzy distance-matrix for rapid
        # computation of the covariance matrix.
        dist = distance_matrix(self.coords,
                               self.coords)
        
        # Compute the covariance between all
        # points in the space.
        self.cov = np.exp(-0.5 * dist ** 2 /
                          self.lamb ** 2)
    
    def plot_covariance_matrix(self):
        """
        Plot the covariance matrix.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.cov, cmap='binary')
        plt.colorbar()
        plt.show()
    
    def compute_eigenpairs(self):
        """
        Find eigenvalues and eigenvectors using Arnoldi iteration.
        """
        eigvals, eigvecs = eigh(self.cov,
                                eigvals=(self.n_points - self.mkl,
                                         self.n_points - 1))
        
        order = np.flip(np.argsort(eigvals))
        self.eigenvalues = eigvals[order]
        self.eigenvectors = eigvecs[:, order]
      
    def generate(self, parameters=None):
        """
        Generate a random field, see
        Scarth, C., Adhikari, S., Cabral, P. H.,
        Silva, G. H. C., & Prado, A. P. do. (2019).
        Random field simulation over curved surfaces:
        Applications to computational structural mechanics.
        Computer Methods in Applied Mechanics and Engineering,
        345, 283–301. https://doi.org/10.1016/j.cma.2018.10.026
        """
        
        if parameters is None:
            self.parameters = np.random.normal(size=self.mkl)
        else:
            self.parameters = np.array(parameters).flatten()
        
        self.random_field = np.linalg.multi_dot((self.eigenvectors, 
                                                 np.sqrt(np.diag(self.eigenvalues)), 
                                                 self.parameters))

    def plot(self, lognormal=True):
        """
        Plot the random field.
        """

        if lognormal:
            random_field = self.random_field
            contour_levels = np.linspace(min(random_field),
                                         max(random_field), 20)
        else:
            random_field = np.exp(self.random_field)
            contour_levels = np.linspace(min(random_field),
                                         max(random_field), 20)

        plt.figure(figsize=(12, 10))
        plt.tricontourf(self.coords[:, 0], self.coords[:, 1],
                        random_field,
                        levels=contour_levels,
                        cmap='plasma')
        plt.colorbar()
        plt.show()

class Matern52(SquaredExponential):
    def assemble_covariance_matrix(self):
        
        '''
        This class inherits from RandomProcess and creates a Matern 5/2 covariance matrix.
        '''
        
        # Compute scaled distances.
        dist = np.sqrt(5)*distance_matrix(self.coords, self.coords)/self.lamb
        
        # Set up Matern 5/2 covariance matrix.
        self.cov =  (1 + dist + dist**2/3) * np.exp(-dist)
