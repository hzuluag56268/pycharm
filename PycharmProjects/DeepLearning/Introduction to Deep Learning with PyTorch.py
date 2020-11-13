'''1  Introduction to PyTorch

'''



......Introduction to PyTorch
import torch

# Create random tensor of size 3 by 3
your_first_tensor = torch.rand(3, 3)

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)




.......Backpropagation by auto-differentiation

# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
f = q * z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))



# Multiply tensors x and y
q = x * y

# Elementwise multiply tensors z with q
f = q * z

mean_f = torch.mean(f)

# Calculate the gradients
mean_f.backward()



......Introduction to Neural Networks




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Instantiate all 2 linear layers
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x






'''2 Artificial Neural Networks'''


    ..........Activation functions
