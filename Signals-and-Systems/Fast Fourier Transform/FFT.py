import numpy as np
import matplotlib.pyplot as plt
import cmath

def fft(x:list):
    n = np.log2(len(x))
    if(n == 0):
        return x
    elif(np.floor(n) == n):
        g = []
        h = []
        for i in range(0,int(len(x)/2)):
            g.append(x[2*i])
            h.append(x[2*i+1])
        g_hat = fft(g)
        h_hat = fft(h)
        x_hat = []
        for i in range(0,int(len(x)/2)):
            x_hat.append(g_hat[i] + h_hat[i]*cmath.exp(-2*np.pi*1j/n))
        for i in range(0,int(len(x)/2)):
            x_hat.append(g_hat[i] - h_hat[i]*cmath.exp(-2*np.pi*1j/n))
        return x_hat

t = np.linspace(0,2*np.pi,int(np.exp2(10.0)))
# x1 = []
# for i in t:
#     if(i>=-np.pi/2 and i <= np.pi/2):
#         x1.append(1.0)
#     else:
#         x1.append(0.0)
# x1 = np.array(x1)
x1 = np.sin(2*t)
# x1_hat = np.array(fft(x1.tolist()))
x1_hat = np.fft.fft(x1)


plt.plot(t,x1)
plt.xlabel('Time(t)')
plt.ylabel('x(t)')
plt.show()

plt.plot(t,x1_hat)
plt.xlabel('Frequency(f)')
plt.ylabel('X(f)')
plt.show()