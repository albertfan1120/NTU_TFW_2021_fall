import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fft import fft


def plot_TFplot(image):
    image = delete_4(image) # for size looking good 
    C = 400
    image = image / np.max(image) * C
    
    plt.figure(0)
    plt.imshow(image, cmap = 'gray', origin = 'lower')
    
    plt.xlabel('Time (Sec)')
    plt.ylabel('Frequency (Hz)')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    
    plt.show()
    

def wdf(x, t, f):
    dt, df = t[1] - t[0], f[1] - f[0]
    T, F = t.shape[0], f.shape[0]
    N = round(1 / (2 * dt * df)) # 1600
    n0, n1 = round(t[0] / dt),  round(t[-1] / dt) # (-720, 719)
    m0, m1 = round(f[0] / df),  round(f[-1] / df) # (-160, 159)
    
    C = np.zeros((F, T))
    m = np.arange(m0, m1+1)
    for n in range(n0, n1 + 1):
        Q = min(n1 - n, n - n0)
        c1 = get_c1(n, Q, x, N, n1+1)
        
        C1 = fft(c1)
        phase = np.exp(1j * 2 * np.pi * Q * m / N)
        C1m = np.abs(2 * C1[m % N] * phase * dt)
        
        C[:, n+n1+1] = C1m
        
    return C
        
        
def get_c1(n, Q, x, N, offset):
    c1 = np.zeros((N), dtype = 'complex_')
    q = np.arange(2*Q + 1)
    
    x_index_real = n + q - Q + offset
    x_index_img = n - q + Q + offset
    c1[:2*Q + 1] = x[x_index_real] * np.conjugate(x[x_index_img])
    
    return c1
    
    
def delete_4(image):
    H, W = image.shape
    new_W = int(W / 4)
    new_image = np.zeros((H, new_W))
    for w in range(new_W):
        new_image[:, w] = image[:, 4*w]
            
    return new_image
    
    
if __name__ == '__main__':
    t = np.arange(-9, 9, 0.0125) # (1440)
    f = np.arange(-4, 4, 0.025)  # (320)
    
    s = np.exp(1j*t**2/10 - 1j*3*t) * (t <= 1) 
    r = np.exp((1j*t**2 / 2) + 1j*6*t) * np.exp(-(t-4)**2 / 10) 
    x = s + r # (1440)
    
   
    time_start = time.time()
    y = wdf(x, t, f)
    time_end = time.time()
    compute_time = time_end - time_start

    print(f"Your WDF consuming time is %.3fs" % compute_time)
    plot_TFplot(y)
