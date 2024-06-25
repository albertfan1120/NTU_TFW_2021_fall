import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

def wavedbc10(x):
    H, W = x.shape
    g = [0.0033, -0.0126, -0.0062,  0.0776, -0.0322, -0.2423, 0.1384, 0.7243,  0.6038,  0.1601]
    h = [0.1601, -0.6038,  0.7243, -0.1384, -0.2423,  0.0322, 0.0776, 0.0062, -0.0126, -0.0033]
    
    V1L, V1H = [], []
    for i in range(H):
        V1L.append(np.convolve(x[i], g)[::2])
        V1H.append(np.convolve(x[i], h)[::2])
    V1L, V1H = np.array(V1L), np.array(V1H)
    
    new_W = V1L.shape[1]
    X1L  = np.zeros((new_W, new_W))
    X1H1 = np.zeros((new_W, new_W))
    X1H2 = np.zeros((new_W, new_W))
    X1H3 = np.zeros((new_W, new_W))
    for i in range(new_W):
        X1L[:, i]  = np.convolve(V1L[:, i], g)[::2]
        X1H1[:, i] = np.convolve(V1L[:, i], h)[::2]
        X1H2[:, i] = np.convolve(V1H[:, i], g)[::2]
        X1H3[:, i] = np.convolve(V1H[:, i], h)[::2]
    
    
    return [X1L, X1H1, X1H2, X1H3]


def iwavedbc10(X1L, X1H1, X1H2, X1H3):
    g1 = [ 0.1601, -0.6038, 0.7243, -0.1384, -0.2423,  0.0322,  0.0776, 0.0062, -0.0126, -0.0033]
    h1 = [-0.0033, -0.0126, 0.0062,  0.0776,  0.0322, -0.2423, -0.1384, 0.7243, -0.6038,  0.1601]
    
    H, W = X1L.shape
    X1L_u, X1H1_u, X1H2_u, X1H3_u = [], [], [], []
    for i in range(H):
        X1L_u.append(X1L[i])
        X1L_u.append(np.zeros((X1L[0].shape[0])))
        X1H1_u.append(X1H1[i])
        X1H1_u.append(np.zeros((X1L[0].shape[0])))
        X1H2_u.append(X1H2[i])
        X1H2_u.append(np.zeros((X1L[0].shape[0])))
        X1H3_u.append(X1H3[i])
        X1H3_u.append(np.zeros((X1L[0].shape[0])))
    X1L_u, X1H1_u  = np.array(X1L_u), np.array(X1H1_u)
    X1H2_u, X1H3_u = np.array(X1H2_u), np.array(X1H3_u)  
    
    V1L = np.zeros((X1L_u.shape[0] + 9, X1L_u.shape[1]))
    V1H = np.zeros((X1L_u.shape[0] + 9, X1L_u.shape[1]))
    for i in range(W):
        V1L[:, i] = np.convolve(X1L_u[:, i], g1)  + np.convolve(X1H1_u[:, i], h1)
        V1H[:, i] = np.convolve(X1H2_u[:, i], g1) + np.convolve(X1H3_u[:, i], h1)
    
    
    V1L_u = np.zeros((V1L.shape[0], V1L.shape[1]*2)) #(531, 261*2)
    V1H_u = np.zeros((V1L.shape[0], V1L.shape[1]*2))
    new_W = V1L.shape[1]
    for i in range(new_W):
        V1L_u[:, 2*i] = V1L[:, i]
        V1H_u[:, 2*i] = V1H[:, i]
    
    X = np.zeros((V1L.shape[0], V1L.shape[0]))
    for i in range(V1L.shape[0]):
        X[i] = np.convolve(V1L_u[i], g1) + np.convolve(V1H_u[i], h1)
    
    
    return X
    
    
def plot(image_list):  
    plt.figure(figsize=(6, 7))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image_list[0], cmap = 'gray')
    plt.gca().set_title('X1L')
    
    plt.subplot(2, 2, 2)
    plt.imshow(image_list[1] > 0.05, cmap = 'gray')
    plt.gca().set_title('X1H1')
    
    plt.subplot(2, 2, 3)
    plt.imshow(image_list[2] > 0.05, cmap = 'gray')
    plt.gca().set_title('X1H2')
    
    plt.subplot(2, 2, 4)
    plt.imshow(image_list[3] > 0.05, cmap = 'gray')
    plt.gca().set_title('X1H3')
        
    plt.show()

def convert(x):
    return ((x - x.min()) * (1/(x.max() - x.min())))


if __name__ == '__main__':
    img = np.array(Image.open("Peppers.bmp").convert('L'))
    img = img.astype('float') / 255.0
    [X1L, X1H1, X1H2, X1H3] = wavedbc10(img)
    
    image_list = [X1L, X1H1, X1H2, X1H3]
    plot(image_list)
    
    X = iwavedbc10(X1L, X1H1, X1H2, X1H3)
    
    X[X < 0] = 0.0
    X[X > 1] = 1.0
    
    plt.imshow(X, cmap = 'gray')
    plt.gca().set_title('cover')
    plt.show()