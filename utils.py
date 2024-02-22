import numpy as np
from tqdm.auto import trange, tqdm
import matplotlib.pyplot as plt
import os

def cvtImg(img):
    img = img - img.min()
    img = (img / img.max())
    return img.astype(np.float32)

def showExamples(x):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        img = cvtImg(x[i])
        plt.imshow(img)
        plt.axis('off')
        
def forwardNoise(x, t, timeBar):
    # returns a pair of noisy images with complete noise at t=0 and non noisy image at t = timeStep
    a = timeBar[t] #baseline image 
    b = timeBar[t+1] # image at next time step
    
    noise = np.random.normal(size = x.shape) #noise mask
    
    a = a.reshape((-1, 1, 1, 1))
    b = b.reshape((-1, 1, 1, 1))
    
    xa = x*(1-a) + noise*a
    xb = x*(1-b) + noise*b
    
    return xa, xb

def generateTimeSteps(num, timeSteps):
    return np.random.randint(0, timeSteps, num)