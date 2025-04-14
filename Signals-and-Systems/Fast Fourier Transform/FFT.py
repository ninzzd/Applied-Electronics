import numpy as np
import matplotlib.pyplot as plt
import cmath

def fft_(x:np.ndarray) -> np.array:
    n = np.log2(x.size)
    if(np.floor(n) == 0):
        return x
    elif(np.floor(n) == n):
        g = []
        h = []
        for i in range(0,int(x.size/2)):
            g.append(x[2*i])
            h.append(x[2*i+1])
        g_hat = fft_(np.array(g))
        h_hat = fft_(np.array(h))
        x_hat = np.zeros(shape=x.size,dtype=np.complex128)
        for i in range(0,int(x.size/2)):
            x_hat[i] = g_hat[i] + h_hat[i]*cmath.exp(-2*np.pi*i*1j/n)
        for i in range(0,int(x.size/2)):
            x_hat[i+int(x.size/2)] = g_hat[i] - h_hat[i]*cmath.exp(-2*np.pi*i*1j/n)
        return x_hat
def fft(x:np.ndarray,t:np.ndarray) -> np.ndarray:
    if(t.size <= 1): 
        return x
    else:
        t_ = t[1] - t[0] # Assuming evenly spaced samples
        x_:np.ndarray = t_*x
        n = x_.size
        p = int(np.log2(n)) + 1
        for i in range(n,int(2**p)):
            x_ = np.append(x_,[0.0])
        print(x_.size)
        x_hat = fft_(x_)
        w = np.linspace(-np.pi/t_,+np.pi/t_,int(2**p),endpoint=False)
        return np.abs(x_hat),w

t = np.linspace(0,2*np.pi,1023,endpoint=False)

x1 = np.zeros_like(t)
# x1[t <= np.pi] = 1
# x1[t > np.pi] = 0
x1 = np.sin(t)

x1_hat,w = fft(x1,t)
print(len(x1_hat))
print(len(w))


plt.plot(t,x1)
plt.xlabel('Time(t)')
plt.ylabel('x(t)')
plt.show()

plt.plot(w,x1_hat)
plt.xlabel('Frequency(f)')
plt.ylabel('X(f)')
plt.show()