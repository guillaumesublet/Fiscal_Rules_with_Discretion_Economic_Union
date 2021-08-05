
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
                beta= 0.8,
                delta = 0.96,
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
        'Returns an array with the first best allocation g_d on the grid'
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
        'Returns an array with the costly discretion allocation g_e on the grid'
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
        alloc = np.empty(len(self.theta_grid))
        theta_c = self.theta_c_F(theta_bar)
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_c:
                alloc[i] = self.discretionary(self.theta_grid[i])
            else:
                alloc[i] = self.discretionary(theta_c)
        return alloc

    # overall
    def disc_costlydisc(self):
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
        alloc = np.empty(len(self.theta_grid))
        theta_n = self.theta_n()
        theta_xp = self.theta_xp(self.theta_bar)
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_n:
                alloc[i] = self.discretionary(self.theta_grid[i])        
            if self.theta_grid[i] > theta_n and self.theta_grid[i] <= theta_xc:    
                alloc[i] = self.costly_disc(self.theta_grid[i])        
            if self.theta_grid[i] > theta_xc:
                alloc[i] = self.costly_disc(theta_xc)        
        return alloc
    
    # overall
    def exempt_costlydisc(self):
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
        alloc = np.empty(len(self.theta_grid))
        theta_x_ = self.theta_x()
        theta_xc_ = self.theta_xc(self.theta_bar)
        for i in range(len(self.theta_grid)):
            if self.theta_grid[i] <= theta_x_:
                alloc[i] = self.costly_disc(theta_x_)
            if self.theta_grid[i] > theta_x_ and self.theta_grid[i] <= theta_xc_:
                alloc[i] = self.costly_disc(self.theta_grid[i])
            if self.theta_grid[i] > theta_xc_:
                alloc[i] = self.costly_disc(theta_xc_)
        return alloc

# Money burning    
    def U(self, g):
        'Utility index'
        return  (g**(1-self.eta))/(1-self.eta)
    
    def U_prime(self, g):
        'Marginal utility'
        return  g**(-self.eta)
    
    def W(self, g):
        'Continuation value'
        return ((self.omega + self.R * (self.T - g))**(1-self.eta))/(1-self.eta)

    def W_prime(self, g):
        'Marginal value of savings'
        return (self.omega + self.R * (self.T - g))**(-self.eta)

    def Delta(self, g):
        'Wedge which is the fiscal need of the government that spending g fulfills'
        A = self.beta * self.delta * self.R * self.W_prime(g)
        B = self.U_prime(g)
        return A/B

    def interp_alloc(self, theta, array):
        'Interpolate an allocation stored in array and evaluated it at theta'
        return interp(self.theta_grid, array, theta)

    # pointwise
    def money_burning(self, theta, array):
        A = theta * self.U(self.interp_alloc(theta, array)) + self.beta * self.delta * self.W(self.interp_alloc(theta, array))
        B = self.grid_min * self.U(array[0]) + self.beta * self.delta * self.W(array[0])
        integral = integrate.quad(lambda x: self.U(self.interp_alloc(x, array)), self.grid_min, theta)
        return A - B - integral[0]

    def tau(self, g, array):
        "utility sanction as a function of spending inferred from money burning as follows tau(g) = t(g^{-1}(g))"
        "WARNING: it is useful only when the allocation has a well defined inverse"
        theta = brentq(lambda x: self.interp_alloc(x, array) - g, self.grid_min, self.grid_max)
        return self.money_burning(theta, array)
    
    def tau_prime(self, g, array):
        if g_grid[i] <= min(array):
            return 0
        if g_grid[i] < max(array) and g_grid[i] > min(array):
            theta = brentq(lambda x: self.interp_alloc(x, array) - g, self.grid_min, self.grid_max)
            return 
        if g_grid[i] >= max(array): #and self.tau(max(array), array) > 10e-6:
            print('tau prime not defined')
            return 10e10
        
    # overall
    def money_burning_schedule(self, array):
        schedule = np.empty(len(self.theta_grid))
        for i in range(len(self.theta_grid)):
            schedule[i] = self.money_burning(self.theta_grid[i], array)
        return schedule
    
    def tau_schedule(self, g_grid, array):
        schedule = np.empty(len(g_grid))
        for i in range(len(g_grid)):
            if g_grid[i] <= min(array):
                schedule[i] = 0
            if g_grid[i] < max(array) and g_grid[i] > min(array):
                schedule[i] = self.tau(g_grid[i], array)
            if g_grid[i] >= max(array): #and self.tau(max(array), array) > 10e-6:
                g = min(g_grid[i], self.discretionary(min(self.theta_bar, 10e4)))
                A = min(self.theta_bar, 10e4) * self.U(g) + self.beta * self.delta * self.W(g)
                B = min(self.theta_bar, 10e4) * self.U(max(array)) + self.beta * self.delta * self.W(max(array))
                schedule[i] = self.money_burning(min(self.theta_bar, 10e4), array) + A - B                
#                g = min(g_grid[i], self.discretionary(self.theta_bar))                
#                A = self.theta_bar * self.U(g) + self.beta * self.delta * self.W(g)
#                B = self.theta_bar * self.U(max(array)) + self.beta * self.delta * self.W(max(array))
#                schedule[i] = self.money_burning(self.theta_bar, array) + A - B
        return schedule

    
# Checks
    def theta_c_exponential(self):
        'Returns theta_p for the exponential distribution'
        assert self.dist == 'exponential', 'Threshold for exponential. Switch distribution to exponential.'
        return (self.beta /(1 - self.beta)) * (1/self.lambda_exp)

    def theta_n_wei(self):
        'Check theta_n for Weibull or truncated Weibull'
        assert self.dist == 'trunc_weibull' or self.dist == 'weibull', 'Threshold for weibull. Switch distribution to weibull.'
        return (self.beta * (1/self.c_weibull) /(1 - self.beta))**(1/self.c_weibull)

    def bound_on_beta(self, theta_l, theta_h, rho):
        theta_eval = np.linspace(theta_l, theta_h, 200)
        upper_b = rho * max(self.invElasticity(theta_eval))
        lower_b = rho * min(self.invElasticity(theta_eval))
        return print(upper_b, 'upper bound', lower_b, 'lower bound')

    def plot_bounds(self, theta_l, theta_h, beta, rho):
        theta_eval = np.linspace(theta_l, theta_h, 200)
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(theta_eval, beta * rho * self.invElasticity(theta_eval), lw=2, alpha=0.6, label='weighted inverse elasticity', color='k')
        plt.axhline(y=1 - beta, linestyle='--', label='1 - $\\beta$', color='gray')
        plt.axhline(y= beta * (rho - 1), linestyle=':', label='$\\beta (\\rho - 1)$', color='gray')
        return plt.show()

    def plot_func_for_theta_xp(self, theta_l, theta_h):
        theta_eval = np.linspace(theta_l, theta_h, 10)
        fig, ax = plt.subplots(figsize=(15,6))
        v_func_for_theta_xp = np.vectorize(self.func_for_theta_xp)
        ax.plot(theta_eval, v_func_for_theta_xp(theta_eval, self.theta_bar), lw=2, alpha=0.6, label='func for theta xc', color='k')
        plt.axhline(y= 0, linestyle=':', label='$0$', color='gray')
        return plt.show()
    
    def plot_func_for_theta_n(self):
        theta_eval = np.linspace(self.grid_min, self.grid_max - 10e-2, 10)
        fig, ax = plt.subplots(figsize=(15,6))
        v_func_for_theta_n = np.vectorize(lambda x: self.beta * self.rho * self.invElasticity(x) - 1 + self.beta)
        ax.plot(theta_eval, v_func_for_theta_n(theta_eval), lw=2, alpha=0.6, label='func for theta e', color='k')
        plt.axhline(y= 0, linestyle=':', label='$0$', color='gray')
        return plt.show()
    
    def plot_func_for_theta_x(self):
        theta_eval = np.linspace(self.grid_min, self.grid_max - 10e-2, 10)
        fig, ax = plt.subplots(figsize=(15,6))
        v_func_for_theta_x = np.vectorize(self.func_for_theta_x)
        ax.plot(theta_eval, v_func_for_theta_x(theta_eval), lw=2, alpha=0.6, label='func for theta xc', color='k')
        plt.axhline(y= 0, linestyle=':', label='$0$', color='gray')
        return plt.show()
    
    def plot_inv_elasticity(self):
        fig, ax = plt.subplots(figsize=(15,6))
        v_inv_elasticity = np.vectorize(self.invElasticity)
        ax.plot(self.theta_grid, v_inv_elasticity(self.theta_grid), lw=2, alpha=0.6, label='inverse of elasticity of 1-F', color='k')
        plt.axhline(y= 0, linestyle=':', label='$0$', color='gray')
        return plt.show()

    def plot_array(self, array):
        fig, ax = plt.subplots(figsize=(15,6))
        ax.plot(self.theta_grid, array, lw=2, alpha=0.6, color='k')
        return plt.show()
        
    def mean_deficit(self):
        E_deficit = integrate.quad(lambda x: self.discretionary(x) * self.f(x), self.grid_min, self.theta_bar)
        return (E_deficit[0] - self.T)/self.T
