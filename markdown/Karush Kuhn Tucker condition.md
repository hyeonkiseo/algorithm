<div style = 'text-align : center'> <font size = '5em'> Convex Optimization - Differentiable problem
</div>

#### Dual problem

- Primal problem 
  $$
  \text{minimize }f(x) \\
  \text{subject to } g_i(x) \leq 0, \ \ i = 1,\cdots,m
  $$
  
  $$
  \text{where } f,g_i \text{are comvex functions on }\R^n
  $$

- Domain : $\mathcal{D} = \textbf{dom }f\cap (\cap^{m}_{i=1}\textbf{dom }g_i)$

- Feasible : a point $x \in \mathcal{D}$ satisfies the constraints $g_i(x) \leq 0$ for $i = 1,\cdots, m$

- Lagrangian L : 
  $$
  L(x,\alpha) = f(x) + \displaystyle \sum^m_{i=1}\alpha_ig_i(x)\ \ \text{where }\alpha_i \geq 0, \forall i
  $$

- (Lagrange) Dual function h
  $$
  h(\alpha) = \underset{x}{\text{inf}} \left\{ f(x) + \displaystyle \sum_{i=1}^m \alpha_ig_i(x)\right\}
  $$

- $h(\alpha) \leq p^{\star} \text{where } p^\star  $ is optimal value of primal problem 
  $$
  \begin{align}
  p^\star &= \underset{x:g_i(x) \leq 0}{\text{inf}} f(x) \\
  &\geq \underset{x:g_i(x) \leq 0}{\text{inf}} \left\{ f(x) + \displaystyle \sum_{i=1}^m \alpha_ig_i(x)\right\} \\ 
  &\geq\underset{x}{\text{inf}} \left\{ f(x) + \displaystyle \sum_{i=1}^m \alpha_ig_i(x)\right\}  = h(\alpha) , \forall\alpha\geq0
  \end{align}
  $$

- Lagrange Dual problem
  $$
  \text{maximize } h(\alpha) \\
  \text{subject to } \alpha \geq 0
  $$

- Weak Duality
  $$
  d^\star \leq p^\star \\
  \text{where }d^\star \text{is optimal value of the dual problem}
  $$

- Strong Duality

$$
d^\star = p^\star \\
\text{When Slater's condition is satisfied} 
$$

- Slater's condition
  $$
  \exist x \in \text{relint}\mathcal{D} \text{       s.t     } g_i(x) < 0, \forall i
  $$

- refined Slater's condition 
  $$
  \exist x \in \text{relint}\mathcal{D} \text{s.t} g_i(x) \leq 0, \forall i , \text{for affine }g_i   
  $$

- Complementary slackness : When Strong Duality holds, $x^\star$ and $\alpha^\star$ are primal and dual optimal point
  $$
  \begin{align}
  f(x^\star) &= h(\alpha^\star) \ \ (\because \text{strong duality}) \\
  &= \underset{x}{\text{inf}} \left\{ f(x) + \displaystyle \sum_{i=1}^m \alpha_i^\star g_i(x)\right\} \\ 
  &\leq f(x^\star) + \displaystyle \sum_{i=1}^m \alpha_i^\star g_i(x^\star) \\ 
  &\leq f(x^\star) \\
  \therefore \displaystyle \sum_{i=1}^m \alpha_i^\star g_i(x^\star) =0 &\rightarrow \alpha_i^\star g_i(x^\star) = 0, \forall i [\text{Complementary Slackness}]
  \end{align}
  $$
  This means that $i$th optimal Lagrange multiplier is zero unless the $i$th constraint is active at the optimum

  

#### KKT optimality conditions (Karash - Kuhn - Tucker)

- Assumption : $f,g_i$ are convex and differentiable

- $x^\star \text{and }\alpha^\star$ are any primal and dual optimal points with zero duality gap 

  $\iff$ Following KKT conditions are satisfied
  $$
  \begin{align}
  \text{i})& g_i(x^\star) \leq 0, \forall i (\text{feasibility}) \\ 
  \text{ii})&\alpha_i^\star \geq 0, \forall i (\text{Lagrange multiplier}) \\
  \text{iii})&\alpha_i^\star g_i(x^\star )= 0 ,\forall i (\text{Complementary Slackness}) \\
  \text{iv)}&\nabla f(x^\star) + \sum_{i=1}^m \alpha_i^\star \nabla g_i(x^\star) = 0 (\text{First derivative have to be 0})
  \end{align}
  $$

