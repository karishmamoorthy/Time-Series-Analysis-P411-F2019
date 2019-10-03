import numpy as np

def pickout(x_array):
    '''
    When you attempt to sum over an array that doesn't have 
    only numbers, it doesn't work
    '''
    x_array = np.array(x_array)
    x_list = list(x_array)
    
    #returns array of booleans stating whether element is nan or not
    oopsie = np.isnan(x_array)
    
    #as you pop numbers out of your list, your list gets shorter... so its your specified index - offset
    offset = 0
    for i in range(len(oopsie)):
        if oopsie[i]:
            a = x_list.pop(i-offset)
            offset += 1
    
    return np.array(x_list)
    
def mean(x_array, picky=False):
    '''
    Takes in an array of numbers for a given random variable
    And returns the mean
    Does the same thing as np.nanmean()
    '''
    if picky:
        x_array = pickout(x_array)
    return sum(x_array)/len(x_array)

def sample_variance(x_array, picky=False):
    '''
    Takes in an array of numbers for a given random variable
    And returns sample variance i.e. assumes that
    we do not have an entire population worth of data
    '''
    if picky:
        x_a = pickout(x_array)
    else:
        x_a = np.array(x_array)
    return sum((x_a - mean(x_a))**2)/(len(x_a) - 1)

def sample_std_dev(x_array, picky=False):
    '''
    Takes in an array of numbers for a given random variable
    And returns sample standard deviation i.e. assumes that
    we do not have an entire population worth of data
    Does the same thing as np.nanstd()
    '''
    return np.sqrt(sample_variance(x_array, picky))

def gaussian_curve(x, mu=0, sigma=1):
    '''
    Takes in x (or array of x) and returns Gaussian(x)
    Defaults to X ~ N(0, 1)
    But user is free to set X ~ N(mu, sigma), 
    where mu is population (or normally distributed sample) mean
    and sigma is population (or normally distributed sample) standard deviation
    
    Note: usually, y = frequency of occurence of x
    '''
    return (1/np.sqrt(2*np.pi))*(1/sigma)*np.e**(-1*((x-mu)**2)/(2*sigma**2))

def integrator(fn, a=0, b=1, dx=1e-4):
    '''
    Takes in a function fn(x)
    and integrates it (using trapezoid integration)
    from a < x < b.
    
    Default value set to a = 0, b = 1 and
    resolution of integration dx = 1e-4
    '''
    n = int((b-a)/dx)
    area = dx*((fn(a) + fn(a))/2)
    for i in range(2, n):
        area += dx*(fn(a + i*dx))
    return area

def confidence_symm(percentage, distribution=gaussian_curve, this_sample_mean=0, this_sample_stdev=1, negative_infinity=-5, dx=1e-4):
    '''
    Returns confidence interval around given mean, for given distribution and
    standard deviation.
    
    this_sample_stdev = population* standard deviation/sqrt(sample_size)
    If not population... then of the larger sample (or set of samples) of which 
    this sample is a part of.
    
    Works only for symmetric unbounded distriubutions... like the normal distribution
    
    Obviously, one cannot perform integration from -infinity to infinity
    So user needs to specify "negative infinity".
    Since usually, we don't need more than 4 significant figures, have set
    default -infinity to -5, because
    gaussian_curve(-5) = 1.4867195147342987e-06
    '''
    alpha = 1.0 - percentage
    
    area = 0
    
    z = negative_infinity
    area += dx*distribution(z_alpha_by_2)
    
    while area < alpha/2:
        z += dx
        area += dx*distribution(z)
    
    z_alpha_by_2 = abs(z)
    
    upper = this_sample_mean + z_alpha_by_2*this_sample_stdev
    lower = this_sample_mean - z_alpha_by_2*this_sample_stdev
    
    return lower, upper 