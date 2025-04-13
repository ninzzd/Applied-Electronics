import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.signal as signal

N = 1024

t = np.linspace(-np.pi,+np.pi,N,endpoint=False)
x = np.sin(100*t) + np.sin(100*t) + np.sin(300*t) + np.sin(400*t)

filter = signal.butter(2,150,'low',analog=True,output='sos')
y = signal.sosfilt(filter,x)

plt.plot(t,x)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.xlim(right = +np.pi,left = -np.pi)
plt.show()

plt.plot(t,y)
plt.ylabel('y(t)')
plt.xlabel('t')
plt.xlim(right = +np.pi,left = -np.pi)
plt.show()
