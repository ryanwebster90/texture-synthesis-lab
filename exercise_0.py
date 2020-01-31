import torch
import time


# Tensors are multi dimensional arrays
# Here are some basic operations we will use in exercise 1 and 2
# You may also type ipython into terminal and import torch to play around

x = torch.tensor([.25,5,-1.1,-5,0]).view(-1,1) # give two dimensions
y = torch.tensor([1.0,-1,0,1.0,0]).view(-1,1) # give two dimensions

# Elementwise multiplication
z = x*y 
print('Elementwise multiplication x*y ')
print(z)


# Dot product  x^t*y
z = torch.mm(x.t(),y)
print('Matrix multiplication with torch.mm() x^t*y ')
print(z)


# Logical indexing
z = x[y>0]
print('Logical indexing x[y>0], (indices 1 and 4)')
print(z)

# create matrices of ones or zeros
z = torch.zeros(3,3)
print('zeros matrix')
print(z)

# create matrices of ones or zeros
z = torch.ones(3,3)
print('ones matrix')
print(z)


# This exercise times GPU vs CPU, to show the massive speedup for linear operators
# Because convolutions are at the heart of deep learning, GPU's are critical
# for modern deep networks to be tractable.

A = torch.randn(4096,1024).cpu()
x = torch.randn(1024,32).cpu()
t0 = time.perf_counter()
for k in range(100):
    y = torch.mm(A,x)

time_per_iteration_cpu = (time.perf_counter() - t0)/100
print('Matrix multiplication time CPU: %.4fs'%(time_per_iteration_cpu))
   
A = A.cuda()
x = x.cuda()
# do iteration to preload gpu
y = torch.mm(A,x)

t0 = time.perf_counter()
for k in range(100):
    y = torch.mm(A,x)
torch.cuda.synchronize()
time_per_iteration_gpu = (time.perf_counter() - t0)/100
print(f'Matrix multiplication time GPU: %.4fs'%(time_per_iteration_gpu)) 
    
    
speedup = time_per_iteration_cpu/time_per_iteration_gpu
print(f'GPU speedup factor = %.4fx'%speedup)