<div style = 'text-align : center'><font size = '5em'> Reproducing Kernel Hilbert Space
</div>

#### Hilbert Space & RKHS

- Hilbert space : a complete linear space where inner product between functions is defined.

- Reproducing Kernel Hilbert space on domain $\mathcal{X}$ : Hilbert space where the evaluation functional $L_x(f) = f(x)$ is bounded 

  *  functional is bounded if there exists a constant $M$ such that $|L(f)| \leq M\|f\|,\ \forall f\in\mathcal{H}$

- Riesz Representation Theorem

  For every bounded linear functional $L$ on a hillbert space $\mathcal{H}$, there exist a unique $\xi_L \in \mathcal{H}$ such that $L(f) = \left< \xi_L , f \right>,\forall f\in \mathcal{H}$. $\xi_L$ is called the $representer$ of $L$ 

  $\rightarrow$ This means that **every evaluation functional in RKHS have its own representator!!**

- There exist $\xi_x \in \mathcal{H}$,  the representer of $L_x(\cdot)$, such that $\left< \xi_x,f \right> = f(x), \forall f \in \mathcal{H}$

- Define $K(x,t) = \xi_x(t)$, called the ***reproducing kernel*** (RK) which is bivariate function.

  $\rightarrow$ We can take this kernel as the representer of point $x$ in any other function in $\mathcal{H}$!!



#### Properties of RK

- nonnegative definite ( = semi positive definite): 

  for every $n$ which is finite, and every $x_1,\cdots,x_n \in \mathcal{X}$, and every $a_1,\cdots,a_n \in \R$
  $$
  \displaystyle \sum_{i=1}^n\sum_{j=1}^n a_ia_j K(x_i,x_j) \geq 0
  $$
  which means that matrix composed of outcomes of $K(x_i,x_j)$ is always n.n.d
  $$
  a'Ka \geq 0,\forall a
  $$
  
- RK is non-negative
  $$
  \sum_{i=1}^n\sum_{j=1}^n a_ia_j K(x_i,x_j) = \sum_{i=1}^n\sum_{j=1}^na_ia_j 
  \left< K(x_i,\cdot), K(x_j,\cdot) \right>  \\ 
  = 
  \left< a_i\sum_{i=1}^nK(x_i,\cdot),\sum_{j=1}^na_j  K(x_j,\cdot) \right> \\ 
  = \left\lVert\sum_{i=1}^nK(x_i,\cdot) \right\rVert^2 \geq 0
  $$

- The Moore- Anronszajn Theroem

  - For every RKHS $\mathcal{H}$ of functions on $\mathcal{X}$, there correspponds a unique RK $K(s,t)$
  - Conversely, for every n.n.d function $K(s,t) $ on $\mathcal{X}$, there corresponds a unique RKHS $\mathcal{H}_K$

  $\rightarrow$ **Reproducing Kernel and RKHS has one to one correspondence. ** 



#### Construct RKHS by function decomposition

- Constructing RKHS by Anronszaijn Theorem

  1) Make space of functions which has form of $f(x) = \sum_m \alpha_m K(x,y_m)$
  2) Define inner product by $\left< K(x,\cdot), K(y,\cdot) \right> = K(x,y)$
  3) Complete that space
  
  

- Mercer-Hilbert-Schmidt Theorem (Eigen-expansion of kernel)

  Mercer Kernel  : a n.n.d function on $\mathcal{X} \times \mathcal{X}$ which satisfy
  $$
  \int_\mathcal{X}\int_\mathcal{X}K^2(x,y)dxdy < \infty \
  $$
  this condition is trivial when $\mathcal{X}$ is compact

- Mercer Kernel can be decomposed with continuous orthonormal eigenfunctions in $L_2$ and eigen value
  $$
  K(x,y) = \displaystyle \sum_{i=1}^\infty \gamma_i \phi_i(x)\phi_i(y)
  $$

- $L_2$ inner product of two univariate function$(\phi_i, \phi_j)$ is defined as $\int \phi_i(x)\phi_j(x)dx = \delta_{ij}$

- It follows that 

  2. 
     $$
     \begin{align}\int_\mathcal{X}\int_\mathcal{X}K^2(x,y)dxdy  &= \sum_i \sum_j \gamma_i \gamma_j \int \phi_i(x) \phi_j(x)dx \int\phi_i(y) \phi_j(y)dy  \\
     &= \sum_i \gamma_i^2 < \infty
     \end{align}
     \\\ \ \ \ \  (1)
     $$
     
  2. 
     $$
     \begin{align}\int_\mathcal{X} K(x,y)\phi_j(x)dx &= \int \sum_i\gamma_i\phi_i(x)\phi_i(y) \phi_j(x) dx \\ 
     &= \sum_i \gamma_i \phi_i(y)\int\phi_i(x)\phi_j(x)dx \\ 
     &= \gamma_j \phi_j(y)
     \end{align}
     \ \ \ \ \ (2)
     $$

  $\rightarrow$ Because of These two properties, mercer kernel is RK in $\mathcal{H}_K$

- $f \in \mathcal{H}_K$ has form  ( by Anronszaijn thm 1)
  $$
  \begin{align}
  f(x) &=  \sum_m \alpha_m K(x,y_m) \\ 
  &= \sum_m \sum_i \alpha_m\gamma_i\phi_i(x)\phi_i(y_m) \\ 
  &=\sum_i\sum_m \alpha_m\phi_i(y_m) \gamma_i \phi_i(x)    \\ 
  &= \sum_i c_i \phi_i(x)  \\
  \text{where }\ c_i &= \gamma_i \sum_m \alpha_m \phi_i(y_m)  \\ 
  &= \sum_m \alpha_m \gamma_i\phi_i(y_m) \\ 
  &= \sum_m \alpha_m (K(x,y_m), \phi_i(x)) \\ 
  &= (\sum_m\alpha_mK(x,y_m),\phi_i(x)) \\ 
  &= (f,\phi_i(x))
  \end{align}
  $$
  

- $\left< K(x,\cdot), f \right>_{\mathcal{H}_K} = f(x)$ implies $ \left< K(x,\cdot), \phi_j \right>_{\mathcal{H}_K}= \sum_i \gamma_i \phi_i(x) \left< \phi_i, \phi_j \right>_{\mathcal{H}_K}  = \phi_j(x)$

  $\rightarrow$ by these properties, **Inner product of $\mathcal{H}_K$**  can be expressed as
  $$
  \begin{align} 
  (\phi_k, \phi_j) &= \left( \phi_k, \sum_i \gamma_i \phi_i\left< \phi_i, \phi_j \right>  \right) \\ 
  &= \sum_i \gamma_i (\phi_k, \phi_i)\left< \phi_i, \phi_j \right> \\
  &= \sum_i \gamma_i \left< \phi_i, \phi_j \right> \int \phi_k(x)\phi_i(x)dx \\ 
  &= \gamma_k \left< \phi_k, \phi_j \right>\\
  \therefore \left< \phi_k, \phi_j \right>_{\mathcal{H}_K} &= \frac{(\phi_k, \phi_j)}{\gamma_k} \\ 
  \\ 
  \therefore \left< f, g \right>_{\mathcal{H}_K} &=  \left< \sum_i^{\infty}c_i\phi_i(x),\sum_j^{\infty}d_j\phi_j(x)\right> \\ 
  &= \left<\sum_i (f,\phi_i)\phi_i(x), \sum_j (g,\phi_j)\phi_j(x) \right> \\ 
  &= \sum_i \frac{(f,\phi_i)(g,\phi_i)}{\gamma_i}  \ \ \ \ \because \text{basis functions are orthonormal} \\ 
  &= \sum_i\frac{c_id_i}{\gamma_i}
  \end{align}
  $$
  and finite norm constraint becomes 
  $$
  \lVert f \rVert ^2= \sum_i^\infty \frac{c_i^2}{\gamma_i} < \infty
  $$
  

- Then is $K(x,\cdot) \in \mathcal{H}_K$?? 
  $$
  K(x,\cdot) = \sum_i \gamma_i\phi_i(x)\phi_i(\cdot)  = \sum_ic_i\phi_i(\cdot) \\ 
  \text{and} \\
  \lVert K(x,\cdot) \rVert^2 = \sum_i\frac{\gamma_i^2\phi_i(x)^2}{\gamma_i} = \sum\gamma_i \phi_i(x)\phi_i(x) = K(x,x) < \infty \\
  \therefore K(x,\cdot) \in \mathcal{H}_K
  $$
   

- Inner product between RK and $f$ 
  $$
  \begin{align} 
  \left< K(x,\cdot), f \right> &= \sum_i \frac{\gamma_i\phi_i(x) c_i}{\gamma_i} \\ 
  &= \sum_i c_i \phi_i(x) = f(x)
  \end{align}
  $$

- Reproducing property
  $$
  \begin{align} 
  \left< K(x,\cdot),K(y,\cdot)  \right>  &= \sum_i \frac{\gamma_i^2 \phi_i(x)\phi_i(y)}{\gamma_i} \\ 
  &= \sum_i \gamma_i \phi_i(x) \phi_i(y) = K(x,y)
  \end{align}
  $$
  or it can be shown by definition of RK
  $$
  K(y,\cdot) \in \mathcal{H}_k \\ 
  \therefore \left< K(x,\cdot), f \right> = f(x) = K(y,x) = K(x,y)
  $$





#### Split of Hilbert space

- If two Hilbert spaces $\mathcal{H}_0$ and $\mathcal{H}_1$ equipped with inner product respectively have the only common element \{0\}, then we define the tensorsum Hilbert space $\mathcal{H} = \{f = f_0 + f_1 : f_0 \in \mathcal{H}_0, f_1 \in \mathcal{H}_1\}$ with inner product $\left<\cdot,\cdot \right> = \left<\cdot,\cdot \right>_0 + \left<\cdot,\cdot \right>_1$  and write $\mathcal{H} = \mathcal{H}_0 \oplus \mathcal{H}_1$ 

- Sum of two n.n.d functions defined on the same domain  is n.n.d
- If $K_0$ is RK for $\mathcal{H}_0$ and $K_1$ is RK for $\mathcal{H}_1$ with $\mathcal{H}_0 \cap \mathcal{H}_1 = \{0\}$, then $K = K_0 + K_1$ is RK for $\mathcal{H} = \mathcal{H}_0 \oplus\mathcal{H}_1$ 

-  If an n.n.d function K in decomposed into two orthogonal n.n.d functions $K_0 \text{ and }K_1$, then $\mathcal{H} = \mathcal{H}_0 \oplus \mathcal{H}_1$, where $\mathcal{H}_0,\mathcal{H}_1$ are RKHS corrsponding to $K_0, K_1$ Because of one to one correspondence of RK and RKHS,

  $\rightarrow$ **We can use each of function space as block** like ANONA



#### Regularization Problems Using RKHS

- we want to use norm of RKHS as penalty

- First variational problem with $J[f] = \|f\|^2_{\mathcal{H}_K}$
  $$
  \underset{f\in \mathcal{H}_K}{min} \left[ \sum_{i=1}^NL(y_i,f(x_i)) + \lambda \|f\|^2_{\mathcal{H}_K}  \right]
  $$

- The solution is finite dimension and has the form of $f(x) = \sum_i^N \alpha_iK(x_i,x)$ where $x_i$ is  observed pts. 

  $\rightarrow$ We can change infinite dimension problems into finite dimension problems with N dimension

  pf) 

  for any functions $\overset{\sim}{g} \in \mathcal{H}_k$ can be decomposed as $g + p$ where $g(x)= \sum_i^N \alpha_iK(x_i,x)$ and $\rho(x) \perp K(x_i,x), \forall i = 1,\cdots, N$ . Then $\rho(x_i) = <K(x_i,x),\rho(x)> = 0$ 

  Thus, $\overset{\sim}{g}(x_i) = g(x_i)$ and $J[\overset{\sim}{g}] = \|\overset{\sim}{g}\|^2_{\mathcal{H}_K} = <g + \rho, g+ \rho> = \|g\|^2 + \|\rho\|^2 \geq \|g\|^2 = J[g]$

  equality holds when $\|\rho\|^2 = 0$ $\rightarrow$ we can find this regardless of loss function!!

- We can change problem as
  $$
  \underset{\alpha}{min}[L(y,K\alpha) + \lambda \alpha'K\alpha]
  $$
  where $K = \{K(x_i,x_j)\}$

- Second variational problam with $J[f] = \|P_1f\|^2_{\mathcal{H}_1}$ where $P_1$ is projection onto $\mathcal{H}_1$
  $$
  \underset{f\in \mathcal{H}_K}{min} \left[ \sum_{i=1}^NL(y_i,f(x_i)) + \lambda \|P_1f\|^2_{\mathcal{H}_1}  \right]
  $$

- The solution is finite dimensional and has the form of $f(x) = \sum_{j=1}^m \beta_j\psi_j(x) + \sum_i^N \alpha_iK_1(x_i,x)$

- We can change problem as
  $$
  \underset{\alpha,\beta}{min} [L(y,T\beta + K_1\alpha) + \lambda \alpha'K_1 \alpha]
  $$
  where $T = \{\psi_j(x_i)\}, K_1 = \{K_1(x_i,x_j)\}$ 

- By using this, We can penalize only the sunset of $\mathcal{H}_k$

