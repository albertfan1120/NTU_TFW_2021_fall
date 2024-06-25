import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def hht(x, t, thr):
    L  = t.shape[0]
    t_ = np.arange(L)
    K  = 30
    y  = x
    
    c = []
    x0 = None
    for n in range(10):
        for k in range(K):
            max_index, min_index = get_max_min(y, L)
            local_max, local_min = y[max_index], y[min_index]
                                                                                            
            f_max = interp1d(max_index, y[max_index], kind = 'cubic', fill_value = "extrapolate")
            f_min = interp1d(min_index, y[min_index], kind = 'cubic', fill_value = "extrapolate")
            e_max, e_min = f_max(t_), f_min(t_) # (1000,)
            
            z = (e_max + e_min) / 2
            h = y - z
            
            max_index, min_index = get_max_min(h, L)
            local_max, local_min = h[max_index], h[min_index]
            f_max = interp1d(max_index, h[max_index], kind = 'cubic', fill_value = "extrapolate")
            f_min = interp1d(min_index, h[min_index], kind = 'cubic', fill_value = "extrapolate")
            e_max, e_min = f_max(t_), f_min(t_) # (1000,)
            
            if (check_local(local_max, local_min) and check_mean(e_max, e_min, thr)) or k == K-1:
                c.append(h)
                break
            else:
                y = h
        
        c_ = np.array(c)
        x0 = x - c_.sum(axis = 0)
        max_index, min_index = get_max_min(x0, L)
        if max_index.shape[0] + min_index.shape[0] <= 3:
            c.append(x0)
            break
        else: 
            y = x0
    
    return np.array(c)
    
    
def check_local(local_max, local_min):
    max_violate = (local_max < 0).sum()
    min_violate = (local_min > 0).sum()
    
    return True if max_violate == min_violate == 0 else False


def check_mean(e_max, e_min, thr):
    mean = (e_max + e_min) / 2
    mean_violate = (mean > thr).sum()
    
    return True if mean_violate == 0 else False


def get_max_min(y, L):
    max_index, min_index = [], []

    for i in range(1, L-1):
        if y[i] >= y[i-1] and y[i] >= y[i+1]:
            max_index.append(i)
        if y[i] <= y[i-1] and y[i] <= y[i+1]:
            min_index.append(i)
    
    
    max_index = np.array(max_index)
    min_index = np.array(min_index)
    
    return max_index, min_index
 
    
def plotIMF(t, y):  
    num = y.shape[0]
    plt.figure(0)
    for i in range(num):
        s = str(num) + '1' + str(i+1)
        plt.subplot(int(s))
        plt.plot(t, y[i])

    plt.show()


if __name__ == '__main__':
    t = np.arange(0, 10, 0.01) # (1000)
    x = 0.2*t + np.cos(2*np.pi*t) + 0.4*np.cos(10*np.pi*t) # (1000)
    thr = 0.2

    y = hht(x, t, thr)
    plotIMF(t, y)
    