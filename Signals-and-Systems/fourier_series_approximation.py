import numpy as np
import matplotlib.pyplot as plt

N = 1024

def convolve(x,y):
    g = []
    for j in range(0,N):
        sum = 0
        for i in range(0,N):
            # Limit of sum integral approximation to continuous-time convolution (circular)
            sum+=x[i]*y[(j-i+int(N/2))%N]/N
        g.append(sum)
    return g

def dirichletKernel(n:int,t):
    d_n = np.zeros_like(t)
    d_n[t != 0] = np.sin(t*(n+0.5))/(np.sin(t/2))
    d_n[t == 0] = (2*n + 1)
    return d_n

def fejerKernel(n:int,t):
    f_n = np.zeros_like(t)
    for i in range(0,n):
        f_n = f_n + dirichletKernel(i,t)
    f_n = f_n/n
    return f_n

t = np.linspace(-np.pi,np.pi,N)
x = np.sin(t)
y = np.sign(t)

# 1 - Square-wave
# x1 = []
# for i in t:
#     if(i>=-np.pi/2 and i <= np.pi/2):
#         x1.append(1.0)
#     else:
#         x1.append(0.0)

# x1 = np.array(x1)


# 2 - Saw-tooth
x1 = []
for i in t:
    if(i>=-np.pi and i<0):
        x1.append(1*(i+np.pi)/np.pi)
    else:
        x1.append(1*(i)/np.pi)
x1 = np.array(x1)
plt.plot(t,x1)
plt.ylabel('x(t)')
plt.xlabel('t')
plt.xlim(right = +np.pi,left = -np.pi)
plt.show()

n = [100]
for i in n:

    d_n = dirichletKernel(i,t)

    f_n = fejerKernel(i,t)
        
    s_n_1 = convolve(x1,d_n)

    s_n_2 = convolve(x1,f_n)

    # plt.plot(t,d_n)
    # plt.ylabel('Dn(t)')
    # plt.xlabel('t')
    # plt.xlim(right = +np.pi,left = -np.pi)
    # plt.show()

    # plt.plot(t,f_n)
    # plt.ylabel('Fn(t)')
    # plt.xlabel('t')
    # plt.xlim(right = +np.pi,left = -np.pi)
    # plt.show()


    plt.plot(t,s_n_1)
    plt.ylabel('Sn(t) [Dirichlet]')
    plt.xlabel('t')
    plt.xlim(right = +np.pi,left = -np.pi)
    plt.show()

    plt.plot(t,s_n_2)
    plt.ylabel('Sn(t) [Fejer]')
    plt.xlabel('t')
    plt.xlim(right = +np.pi,left = -np.pi)
    plt.show()