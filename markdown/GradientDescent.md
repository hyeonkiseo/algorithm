<div style  = 'text-align : center'><font size = '5em'> Convex Optimization - Gradient Descent</div>

### Optimization

- Find value of decision variables that make object function have minimum(maximum) value.

- In machine learning, object function becomes Cost function which have to be minimized.

- e.g. : MSE, Entrophy, etc.



### Gradient

- $\nabla f(x)$ : Vector whose components are partial derivatives of $f$ at some point $x$

$$
\nabla f(x)  = \left( \frac{\partial f}{\partial x_1} , \cdots, \frac{\partial f}{\partial x_p} \right)^T
$$

- Which can be interpreted as **direction and magnitude in which  the function moves  ** 
- If we want to minimize objective function, We have to update variable in **negative direction of derivatives!!**

![Gradient Descent](/Users/hyeonki/Downloads/Gradient Descent.jpeg)



### Algorithm

- Objective : $\underset{\theta}{\text{min}}J(\theta)$
- Parameter : $\theta = [\theta_0. \theta_1  \cdots \theta_p]$
- Algorithm 
  1. Set $t = 0$ and choose the learning rate(step size) $\alpha$
  2. Compute $\nabla J(\theta)$
  3. Set $\theta^{(t+1)} := \theta^{(t)} - \alpha \nabla J(\theta)$
  4. Repeat 2~3 until $\nabla J(\theta) = 0$



### Example - linear regression

- Functions
  $$
  J(\theta) = \frac{1}{2m}\displaystyle \sum^m_{i=1}(\hat{y}_i - y_i)^2 \\ 
  \hat{y}_i = \theta_0 + \theta_1x_{i1} + \cdots + \theta_px_{ip} \\
  \\
  $$

-  where $j \neq 0$
  $$
  \begin{align}
  \frac{\partial J}{\partial \theta_j} &= \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum(\hat{y}_i - y_i)^2 \\ 
  &= 2 \dotproduct \frac{1}{2m} \sum (\hat{y}_i - y_i) \frac{\partial }{\partial \theta_j}(\hat{y}_i - y) \\ 
  &= \frac{1}{m}\sum(\hat{y}_i - y)x_{ij}
  \\
  \frac{\partial J}{\partial \theta_0} &= \frac{1}{m}\sum(\hat{y}_i - y)
  
  \end{align}
  $$

- Repeat until converge
  $$
  \\
  \theta_0 := \theta_0 - \alpha\frac{1}{m}\sum(\hat{y}_i - y) \\
  \theta_j := \theta_1 - \alpha\frac{1}{m}\sum(\hat{y}_i - y)x_{ij}
  $$



### Learning rate 

- If $\alpha$ is too high : It can diverge

- If $\alpha$ is too low : It is very slow to be converge

  ![diverge](/Users/hyeonki/Downloads/diverge.jpeg)



### Steepest Descent

- Disadvantage of GD : converge is very slow

![slowconverge](/Users/hyeonki/Downloads/slowconverge.jpeg)

- If direction is not changed, there is no need to slowly update parameters

   $\rightarrow$ recalculate learning rate every step!!

- Algorithm

  1. Set $t = 0$ 
  2. Compute $\nabla J(\theta)$
  3. Compute learning rate$\alpha_i = \underset{\alpha \geq 0}{\text{argmin}} f(x_i - \alpha\nabla J(\theta))$ 
  4. Set $\theta^{(t+1)} := \theta^{(t)} - \alpha \nabla J(\theta)$
  5. Repeat 2~4 until $\nabla J(\theta) = 0$

![steepestGD](/Users/hyeonki/Downloads/steepestGD.jpeg)