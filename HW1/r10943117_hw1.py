import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.fft import fft

def recSTFT(x, t, f, B):
    T, F = t.shape[0], f.shape[0]
    dt = t[1] - t[0]
    df = f[1] - f[0]
    N = round(1 / (dt * df)) 
    Q = int(B / dt)
    
    f_bound = int(F/2)
    X = np.zeros((F, T))
    for n in range(T):
        # Get x1
        x1 = get_x1(n, Q, x, N) 
        m = np.arange(F) 
        
        # Compute X1m
        phase = np.exp(1j * 2 * np.pi * (Q-n) * m / N) 
        X1 = fft(x1)
        X1m = np.abs(X1 * phase * dt)
        
        # Convert X1m into X
        index = np.arange(f_bound, -f_bound, -1)
        f_axis = index - f_bound
        X[f_axis, n] = X1m[index % N]
        
    return X
        

def get_x1(n, Q, x, N):
    x1 = np.zeros((N)) # 100
    for q in range(2*Q + 1):
        index = n - Q + q
        if 0 <= index < x.shape[0]:
            x1[q] = x[index]
    
    return x1

    

def plot_TFplot(image):
    C = 800
    image = image / np.max(image) * C
    
    plt.imshow(image, cmap = 'gray', origin = 'lower')
    plt.xlabel('Time (Sec)')
    plt.ylabel('Frequency (Hz)')
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.show()


if __name__ == '__main__':
    t = np.arange(0, 30, 0.1)
    f = np.arange(-5, 5, 0.1)
    x1 = np.cos(2 * np.pi * t[:100])
    x2 = np.cos(7 * np.pi * t[100:200])
    x3 = np.cos(4 * np.pi * t[200:])
    x = np.concatenate((x1, x2, x3))
    B = 0.5
    
    time_start = time.time()
    y = recSTFT(x, t, f, B)
    time_end = time.time() 
    compute_time = time_end - time_start
    
    print(f"Your recSTFT consuming time is %.3fs" % compute_time)
    plot_TFplot(y)
    
    