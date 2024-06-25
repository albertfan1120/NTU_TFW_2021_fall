import wave
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy.fft import fft


def plot_TFplot(image):
    image = copy_10(image)
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
    
    
def getWave(path):
    wavefile = wave.open(path, 'rb')
    num_frame = wavefile.getnframes()

    str_data = wavefile.readframes(num_frame)
    wave_data = np.frombuffer(str_data, dtype=np.int16)
    
    delete = np.arange(0, 141804, 2) 
    wave_data = np.delete(wave_data, delete)
    
    return  wave_data


def Gabor(x, tau, t, f, sgm):
    dtau, dt, df = tau[1] - tau[0], t[1] - t[0], f[1] - f[0]
    C, F = t.shape[0], f.shape[0]
    N = round(1 / (dtau * df)) # 44100
    Q = round(1.9143 / (((sgm ** 0.5) * dtau))) # 5659
    S = round(dt / dtau) # 441

    
    X = np.zeros((F, C))
    w = get_w(Q, N, sgm, dtau)
    for n in range(C):
        # get x1
        x1 = get_x1(n, Q, x, N, S, w)
        
        X1 = fft(x1)
        
        m = f
        phase = np.exp(1j * 2 * np.pi * (Q-n*S) * m / N)
        X1m = np.abs(X1[m % N] * phase * dtau)
        X[:, n] = X1m
        
    return X
        
        
def get_x1(n, Q, x, N, S, w):
    x1 = np.zeros((N)) 
    for q in range(2*Q + 1):
        x_index = n * S - Q + q
        if 0 <= x_index < x.shape[0]:
            x1[q] = x[x_index] 
    
    return x1 * w
    
    
def get_w(Q, N, sgm, dtau):
    w = np.arange(2*Q + 1)
    zero = np.zeros(N - 1 - 2*Q)
    w_index = (Q - w) * dtau
    
    w = np.exp(-sgm * math.pi * (w_index ** 2)) * (sgm ** 0.25)
    w = np.concatenate((w, zero))
    
    return w

def copy_10(image):
    H, W = image.shape
    new_image = np.zeros((H, 10*W))
    for w in range(W):
        for i in range(10):
            new_image[:, 10*w+i] = image[:, w]
            
    return new_image
    
if __name__ == '__main__':
    wave_data = getWave('./Chord.wav')
    tau = np.arange(0, 1.6077, 1 / 44100)  # (70560)
    t = np.arange(0, np.max(tau), 0.01) # (160)
    f = np.arange(20, 1000, 1)  # (980,)
    sgm = 200
    
    time_start = time.time()
    y = Gabor(wave_data, tau, t, f, sgm)
    time_end = time.time()
    compute_time = time_end - time_start
    print(f"Your Gabor consuming time is %.3fs" % compute_time)
    plot_TFplot(y)
