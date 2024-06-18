import torch
import gpytorch
kernel= gpytorch.kernels.RBFKernel(ard_num_dims=1, batch_shape=torch.Size([1]))
from torch.autograd.functional import jacobian
def kernel_output(x1, x2):   

    output = kernel(x1, x2).evaluate()

    output = torch.sum(output, dim=1)
    return output
x1 = torch.linspace(0, 1, 3).reshape(-1, 1)
x2 = torch.linspace(0, 1, 3).reshape(-1, 1)

def return_jacobian(x, y):
    inputs = (x, y)
    jac=jacobian(kernel_output, inputs, create_graph=True)
    jac_1= torch.sum(jac[0], dim=1)
    return jac_1
def hessian(x,y):
    inputs = (x, y)
    hess= jacobian(return_jacobian, inputs)
    hessian= hess[1].permute(0,1,3,2,4)
    hessian=hessian.diagonal(dim1=-2, dim2=-1)
    hessian= hessian.permute(0,3,1,2)
    return hessian
# jac=jacobian(exp_adder, inputs, create_graph=True)
# print(jac[0])
inputs = (x1, x2)
jac1=return_jacobian(inputs[0], inputs[1])
hess= hessian(inputs[0], inputs[1])
hess= hess.squeeze()

print(hess.shape)