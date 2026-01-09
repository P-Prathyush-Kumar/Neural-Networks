import numpy as np 
arr1 = np.array(range(1,4))
arr2 = np.array([[1,2],[3,4],[5,6]])
b = np.array([8,4])

z = np.dot(arr1,arr2) + b
print("output: ",z)

def e(x):
    return np.exp(x)

def sigmoid(x):
    return 1 / (1 + e(-x))
def Tanh(x):
    return ((e(x) - e(-x))/(e(x) + e(-x)))

print(sigmoid(z))
print(Tanh(z))

# ----- LOSS -----
y = np.array([1, 0])   # target

loss = - (y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z)))
print("loss:", loss)

# ----- BACKWARD -----
dz = sigmoid(z) - y          
dW = np.outer(arr1, dz)      
db = dz                      

print("dz:", dz)
print("dW:", dW)
print("db:", db)

