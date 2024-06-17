import torch
import gpytorch
kernel= gpytorch.kernels.RBFKernel(ard_num_dims=3)

# Define inputs (batches of points)
x1 = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
x2 = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

x = torch.cat([x1, x2], dim=1)
# Evaluate the kernel on the inputs
def kernel_output(x):   
    # Calculate the midpoint
    midpoint = x.size(1) // 2

    # Select the first part
    x1 = x[0,:midpoint].reshape(1,-1)

    # Select the second part
    x2 = x[0,midpoint:].reshape(1,-1)

    output = kernel(x1, x2).evaluate()
    return output

# compute hessian of the kernel output

def hessian(x):
    hess=torch.autograd.functional.hessian(kernel_output, x)
    return hess


hess = hessian( x)
print(hess)


