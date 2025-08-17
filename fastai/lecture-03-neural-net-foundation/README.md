# Lesson 03- Neural net foundations

## References
- Lecture link: [https://course.fast.ai/Lessons/lesson3.html]
- Github link: [https://github.com/fastai/fastbook/blob/master/clean/04_mnist_basics.ipynb]
- How does a neural network really work: [https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work]


## How to do a fast.ai lesson

- Watch Lecture
- Run notebook and experiment
- Reproduce results [use github.com/fastai/fastbook repo]
- Repeat with different dataset

## Which image models are best

Mostly we care about three things in an image model:
- How fast are they
- How much memory they use
- How accurate are they

Paperspace Gradient for running notebook and experiment.

Kaggle notebook- how does a neural network actually work


## How do we fit a function to data

A general quadratic function:

```
def quad(a,b,c,x): return a*x**2 + b*x + c

quad(3,2,1, 1.5)

from functools import partial
def mk_quad(a,b,c): return partial(quad, a, b, c)

f = mk_quad(3,2,1)
f(1.5)
```

```
from numpy.random import normal,seed,uniform
np.random.seed(42)
def noise(x, scale): return normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x+(1+noise(x,mult))+ noise(x,add)

x = torch.linspace(-2,2, steps=20)[:,None] // this creates a vector/tensor
y = add_noise(f(x), 0,3, 1.5) // y values is f(x) with reandom noise with it
plt.scatter(x,y);
```

Interacting with fit functions:
```
from ipywidgets import interact

@interact(a=1.5, b=1.5, c=1.5)
def plot_quad(a,b,c):
    plt.scatter(x,y)
    plot_function(mk_quad(a,b,c), ylim=(-3,12))
```

For the large models, with billions of parameters it is not possible to fit the params values like this manually.


## Loss functions

- Loss function determines how good or bad a model performs against actual data


A simplest form of loss function- mean square error:
```
def mse(preds, acts): return ((preds-acts)**2).mean()
```

```
@interact(a=1.5, b=1.5, c=1.5)
def plot_quad(a, b, c):
    f = mk_quad(a,b,c)
    plt.scatter(x,y)
    loss=mse(f(x), y)
    plot_function(f, ylim=(-3,12), title=f"MSE: {loss:.2f}")
```

This is a function that takes co-efficients of quadratic and returns a loss. It's wrapped up by PyTorch machinery, that allows you to do things like calculate derrivatives
```
def quad_mse(params):
    f = mk_quad(*params)
    return mse(f(x), y)
```

```
quad_mse([1.5, 1.5, 1.5])
```

Rank 1 tensor
```
#rank 1 tensor
abc = torch.tensor([1.5, 1.5, 1.5])
abc.requires_grad_() // Telling Pytorch to calculate the gradient of these numbers
```

```
loss = quad_mse(abc)
loss
loss.backgoud() // It will not show any output, instead it will generate a property for abc called grad. That will contain the tensor indicating the loss
abc.grad
```

mse =mean-square-error
```
with torch.no_grad():
    abc -= abc.grad*0.01
    loss = quad_mse(abc)

print(f'loss={loss:.2f}')
```

automate the search
```
for i in range(5):
    loss = quad_msr(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.01
    print(f'step={i}; loss={loss:.2f})
```

This method is called gradient descent 

## Rectified Linear function

```
def rectified_linear(m,b,x):
    y = m*x+b
    return torch.clip(y, 0.)
```

```
plot_function(partial(rectified_linear, 1, 1))
```

- relu: Rectified Linear Units
```
@interact(m=1.5, b=1.5)
def plot_relu(m,b):
    plot_function(partial(rectified_linear, m, b), ylim=(-1,4))
```

Double relu
```
def double_relu(m1, b1, m2, b2, x):
    return rectified_linear(m1,b1,x) + rectified_liener(m2,b2,x)
```

```
@interact(m1=1.5,b1=1.5,m2=1.5,b2=1.5)
def plot_double_relu(m1,b1, m2, b2):
    plot_function(partial(double_relu, m1, b1, m2, b2), ylim=(-1,6))
```

It's inconvenient to do many relus together. For efficient execution of relu, we need to do matrix multiplication.
The GPUs are good at matrix multiplication. GPUs have special cores called tensor cores for matrix multiplication. 

## Build a regression model in spreadsheet from Titanic data