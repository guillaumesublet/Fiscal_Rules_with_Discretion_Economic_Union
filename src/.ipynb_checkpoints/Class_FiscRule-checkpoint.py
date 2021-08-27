
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import brentq
from interpolation import interp
from scipy.stats import expon       # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html#scipy.stats.expon
from scipy.stats import pareto      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html#scipy.stats.pareto
from scipy.stats import weibull_min # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html#scipy.stats.weibull_min
from scipy.stats import gamma       # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
from scipy.stats import gengamma    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.gengamma.html#scipy.stats.gengamma

class FiscRule ():
    def __init__(self,
                R = 1.04,
                T = 1,
                beta= 0.8,               # 1-beta is the degree of present bias
                delta = 0.96,            # discount factor of the population
                rho = 1/0.8,
                omega = 0.98,
                eta = 2,
                dist = 'exponential',
                lambda_exp = 3,          # parameter of the exponential distribution
                gamma = 4,               # parameter of the tail of the Pareto distribution
                a_mixture  = 0.375,      # parameter of the distribution F_a described in the text
                c_weibull = 0.5,         # parameters of the Weibull distribution
                scale_weibull = 1,
                loc_weibull = 1,
                a_gamma = 0.5,           # parameters of the gamma distribution
                scale_gamma = 0.5,
                theta_bar = np.inf,      # truncation of the upper tail of the distribution
                grid_min = 1,            # lower bound of the fiscal needs $\underline\theta$
                grid_max = 3,            # highest fiscal need on the graph
                grid_size = 200):        # number of grid points
        self.R, self.T, self.beta, self.delta, self.rho = R, T, beta, delta, rho
        self.omega, self.eta = omega, eta
        self.dist, self.lambda_exp, self.gamma, self.a_mixture = dist, lambda_exp, gamma, a_mixture
        self.c_weibull, self.scale_weibull, self.loc_weibull = c_weibull, scale_weibull, loc_weibull
        self.a_gamma, self.scale_gamma = a_gamma, scale_gamma
        self.theta_bar = theta_bar
        self.grid_min, self.grid_max = grid_min, grid_max
        self.theta_grid = np.linspace(grid_min, grid_max, grid_size)

# Distributions
    def F(self, theta):
        'cdf of the distribution of fiscal needs'
        if self.dist == 'exponential':
            F = expon.cdf(theta, scale=1/self.lambda_exp, loc=self.grid_min)
        if self.dist == 'pareto':
            F = pareto.cdf(theta, self.gamma, loc=self.grid_min-1)
        if self.dist == 'trunc_pareto':
            integral = integrate.quad(self.f, self.grid_min, theta)
            F = integral[0]
        if self.dist == 'mixture':
            F = 1 - (np.exp(- self.lambda_exp * (theta - self.grid_min))**self.a_mixture) * (((self.grid_min/theta)**(self.gamma))**(1-self.a_mixture))
        if self.dist == 'trunc_mixture':
            F = (1 - (np.exp(- self.lambda_exp * (theta - self.grid_min))**self.a_mixture) * (((self.grid_min/theta)**(self.gamma))**(1-self.a_mixture)))/ (1 - (np.exp(- self.lambda_exp * (self.theta_bar - self.grid_min))**self.a_mixture) * (((self.grid_min/self.theta_bar)**(self.gamma))**(1-self.a_mixture)))
        if self.dist == 'weibull':
            F = weibull_min.cdf(theta, self.c_weibull, loc=self.loc_weibull, scale=self.scale_weibull)
        if self.dist == 'trunc_weibull':
            integral = integrate.quad(self.f, self.grid_min, theta)
            F = integral[0]
        if self.dist == 'gamma':
            F = gamma.cdf(theta, self.a_gamma, scale=self.scale_gamma)            
        if self.dist == 'gengamma':
            F = gengamma.cdf(theta, self.a_gamma, self.c_weibull, loc=self.grid_min)
        return F

    def f(self, theta):
        'pdf of the distribution of fiscal needs'
        if self.dist == 'exponential':
            f = expon.pdf(theta, scale=1/self.lambda_exp, loc=self.grid_min)
        if self.dist == 'pareto':
            f = pareto.pdf(theta, self.gamma, loc=self.grid_min-1)
        if self.dist == 'trunc_pareto':
            if theta > self.theta_bar:
                f = 0
            else:         
                f = pareto.pdf(theta, self.gamma, loc=self.grid_min-1)/pareto.cdf(self.theta_bar, self.gamma, loc=self.grid_min-1)            
        if self.dist == 'weibull':
            f = weibull_min.pdf(theta, self.c_weibull, loc = self.loc_weibull, scale = self.scale_weibull)
        if self.dist == 'trunc_weibull':
            f = weibull_min.pdf(theta, self.c_weibull, loc = self.loc_weibull, scale = self.scale_weibull)/(weibull_min.cdf(self.theta_bar, self.c_weibull, scale = self.scale_weibull) - weibull_min.cdf(self.grid_min, self.c_weibull, scale=self.scale_weibull))
        if self.dist == 'gamma':
            f = gamma.pdf(theta, self.a_gamma, scale = self.scale_gamma)
        if self.dist == 'gengamma':
            f = gengamma.pdf(theta, self.a_gamma, self.c_weibull, loc = self.grid_min) 
        if self.dist == 'mixture':
            f = (self.a_mixture*self.lambda_exp + (1-self.a_mixture)*self.gamma/theta) * (1-self.F(theta))
        if self.dist == 'trunc_mixture':
            f = (self.a_mixture*self.lambda_exp + (1-self.a_mixture)*self.gamma/theta) * ((np.exp(- self.lambda_exp * (theta - self.grid_min))**self.a_mixture) * (((self.grid_min/theta)**(self.gamma))**(1-self.a_mixture))) / (1 - (np.exp(- self.lambda_exp * (self.theta_bar - self.grid_min))**self.a_mixture) * (((self.grid_min/self.theta_bar)**(self.gamma))**(1-self.a_mixture)))
        return f
    
    def CTE(self, theta, theta_bar):
        'Conditional tail expectation'
        integral = integrate.quad(lambda x: x * self.f(x), theta, theta_bar)
        return integral[0]/(1 - self.F(theta))
        
    def invElasticity(self, theta):
        'Inverse of the elasticity of the tail 1-F'
        return (1 - self.F(theta))/(theta * self.f(theta))

    def mean_fiscal_need(self):
        'Returns the expecation of the fiscal needs'
        if self.dist == 'exponential':
            E_theta = expon.mean(scale=1/self.lambda_exp, loc=self.grid_min)
        if self.dist == 'pareto':
            E_theta = pareto.mean(self.gamma, loc=self.grid_min-1)
        if self.dist == 'trunc_weibull' or 'mixture':
            E_theta = integrate.quad(lambda x: x * self.f(x), self.grid_min, self.theta_bar)
            E_theta = E_theta[0]                    
        if self.dist == 'weibull':
            E_theta = weibull_min.mean(self.c_weibull, loc = self.loc_weibull, scale = self.scale_weibull)
        if self.dist == 'gamma':
            E_theta = gamma.mean(self.a_gamma, scale = self.scale_gamma)            
        if self.dist == 'gengamma':
            E_theta = gengamma.mean(self.a_gamma, self.c_weibull, loc = self.grid_min)                        
        return E_theta
    
    def mean_deficit(self):
        'Returns the expected deficit (in percentage of fiscal revenues) from the discretionary allocation'
        E_deficit = integrate.quad(lambda x: self.discretionary(x) * self.f(x), self.grid_min, self.theta_bar)
        return (E_deficit[0] - self.T)/self.T
    
# Thresholds
    def theta_p(self, theta_bar):
        'Returns theta_p the threshold above which the prohibitive sanctions are constraining'
        return brentq(lambda x: self.beta * self.CTE(x, theta_bar) - x, self.grid_min, self.grid_max - 10e-2)
        
    def theta_n(self):
        'Returns theta_n the threshold above which the non-prohibitive sanctions are meted out'
        return brentq(lambda x: self.beta * self.rho * self.invElasticity(x) - 1 + self.beta, self.grid_min, self.grid_max - 10e-2)
            
    def func_for_theta_xp(self, theta, theta_bar):
        'Function the root to which is theta_xp'
        integral = integrate.quad(lambda x: (self.rho * x * self.invElasticity(x) + (1-self.rho) * (x - theta)) * self.f(x), theta, theta_bar)
        return integral[0]/(1 - self.F(theta)) - self.rho * theta * self.invElasticity(theta)
    
    def theta_xp(self, theta_bar):
        'Returns theta_xp the threshold above which the non-prohibitive sanctions are meted out and there were expemtions'
        return brentq(lambda x: self.func_for_theta_xp(x, theta_bar), self.grid_min, self.grid_max - 10e-2)
    
    def func_for_theta_x(self, theta):
        'Function the root to which is theta_x'
        integral = integrate.quad(lambda x: (x - theta) * self.f(x), self.grid_min, theta)
        return theta * self.invElasticity(theta) * self.F(theta) - (integral[0]/self.rho) - theta *((1 - self.beta - self.beta * self.rho * self.invElasticity(theta))/(1 - self.beta - self.beta * (self.rho - 1)))
    
    def theta_x(self):
        'Returns theta_x the threshold below which the non-prohibitive sanctions are not meted out, i.e. exemption'
        return brentq(lambda x: self.func_for_theta_x(x), self.grid_min, self.grid_max*3/5)

    def theta_A4(self):
        'Returns theta_A4 the threshold in Assumption A4 when the distribution is exponential'
        assert self.dist == 'exponential', 'Distribution is not exponential and the formula for theta_A4 is based on the exponential distribution'
        return (1/self.lambda_exp) * (1/(1-self.beta))
     
# Allocations
    # pointwise definition
    def discretionary(self, theta):
        'Discretionary allocation g_d evaluated at theta'
        A = self.omega + self.R * self.T
        B = ((self.beta * self.delta * self.R)/theta)**(1/self.eta)
        return A / (self.R + B)

    # overall
    def discretionary_alloc(self):
        'Returns an array with the discretionary allocation g_d on the grid'
        alloc = np.empty(len(self.theta_grid))
        for i in range(len(self.theta_grid)):
            alloc[i] = self.discretionary(self.theta_grid[i])
        return alloc
    
    # pointwise definition
    def first_best(self, theta):
        'First best allocation g^* evaluated at theta'
        A = self.omega + self.R * self.T
        B = ((self.delta * self.R)/theta)**(1/self.eta)
        return A / (self.R + B)

    # overall
    def state_contingent_alloc(self):
        'Returns an array with the first best allocation g^* on the grid'
        alloc = np.empty(len(self.theta_grid))
        for i in range(len(self.theta_grid)):
            alloc[i] = self.first_best(self.theta_grid[i])
        return alloc

    # pointwise definition
    def costly_disc(self, theta):
        'Costly discretion allocation g_n evaluated at theta'
        A = self.omega + self.R * self.T
        B = ((self.beta * self.delta * self.R)/theta)**(1/self.eta)
        C = ((self.beta/(1-self.beta * self.rho)) * (self.rho * self.invElasticity(theta) + 1 - self.rho))**(-1/self.eta)
        return A / (self.R + B * C)
    
    # overall
    def costly_disc_alloc(self):
        'Returns an array with the costly discretion allocation g_n on the grid'
        alloc = np.empty(len(self.theta_grid))
        for i in range(len(self.theta_grid)):
            alloc[i] = self.costly_disc(self.theta_grid[i])
        return alloc
    
    # pointwise definition
    def tight_cap(self):
        'Returns g_c'
        assert self.beta * self.rho * max(self.invElasticity(self.theta_grid)) <= self.beta * self.rho - self.beta, 'Deficit bias is not high.'
        A = self.omega + self.R * self.T
        if self.dist == 'exponential':
            E_theta = expon.mean(scale = 1/self.lambda_exp, loc = self.grid_min)
        if self.dist == 'pareto':
            E_theta = pareto.mean(self.gamma, loc=self.grid_min-1)
        if self.dist == 'weibull':
            E_theta = weibull_min.mean(self.c_weibull, loc = self.loc_weibull, scale = self.scale_weibull)
        if self.dist == 'gamma':
            E_theta = gamma.mean(self.a_gamma, scale = self.scale_gamma)            
        if self.dist == 'gengamma':
            E_theta = gengamma.mean(self.a_gamma, self.c_weibull, loc = self.grid_min)                        
        if self.dist == 'trunc_weibull' or 'mixture' or 'trunc_mixture':
            E_theta_temp = integrate.quad(lambda x: x * self.f(x), self.grid_min, self.theta_bar)
            E_theta = E_theta_temp[0]
        B = ((self.delta * self.R)/E_theta)**(1/self.eta)        
        return A / (self.R + B)

    # overall
    def tight_cap_alloc(self):
        'Returns the tight cap allocation, constant at g_c, on a grid'
        alloc = np.empty(len(self.theta_grid))
        for i in range(len(self.theta_grid)):
            alloc[i] = self.tight_cap()
        return alloc    

    # overall
    def disc_cap_alloc(self, theta_bar):
        'Returns the discretion and cap allocation g_d^p'
        alloc = np.empty(len(self.theta_grid))
        theta_p = self.theta_p(theta_bar)
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_p:
                alloc[i] = self.discretionary(self.theta_grid[i])
            else:
                alloc[i] = self.discretionary(theta_p)
        return alloc

    # overall
    def disc_costlydisc(self):
        'Returns the discretion and costly discretion allocation g_d^n'
        alloc = np.empty(len(self.theta_grid))
        theta_n = self.theta_n()
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_n:
                alloc[i] = self.discretionary(self.theta_grid[i])
            else:
                alloc[i] = self.costly_disc(self.theta_grid[i])
        return alloc

    # overall    
    def disc_costlydisc_prohib(self):
        'Returns the discretion, costly discretion and cap allocation g_d^{np}'
        alloc = np.empty(len(self.theta_grid))
        theta_n = self.theta_n()
        theta_xp = self.theta_xp(self.theta_bar)
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_n:
                alloc[i] = self.discretionary(self.theta_grid[i])        
            if self.theta_grid[i] > theta_n and self.theta_grid[i] <= theta_xp:    
                alloc[i] = self.costly_disc(self.theta_grid[i])        
            if self.theta_grid[i] > theta_xp:
                alloc[i] = self.costly_disc(theta_xp)        
        return alloc
    
    # overall
    def exempt_costlydisc(self):
        'Returns the exemption and costly discretion allocation g_x^n'
        alloc = np.empty(len(self.theta_grid))
        theta_x = self.theta_x()
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_x:
                alloc[i] = self.costly_disc(theta_x)
            else:
                alloc[i] = self.costly_disc(self.theta_grid[i])
        return alloc

    # overall
    def exempt_costlydisc_prohib(self):
        'Returns the exemption, costly discretion and cap allocation g_x^p'        
        alloc = np.empty(len(self.theta_grid))
        theta_x_ = self.theta_x()
        theta_xp_ = self.theta_xp(self.theta_bar)
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_x_:
                alloc[i] = self.costly_disc(theta_x_)
            if self.theta_grid[i] > theta_x_ and self.theta_grid[i] <= theta_xp_:
                alloc[i] = self.costly_disc(self.theta_grid[i])
            if self.theta_grid[i] > theta_xp_:
                alloc[i] = self.costly_disc(theta_xp_)
        return alloc
