<div style = 'text-align : center'><font size ='5em'>  Real analysis - introduction
</div>

### Limitation of Riemann's integral

1) Riemann integrable function space is not have one to one correspondence with $\ell^2(\Z)$ 

   - By Parseval's identity, $L_2$ space inner product is same as $\ell^2(Z)$ space inner produc
     $$
     \displaystyle \sum_{n=-\infty}^\infty |a_n|^2 = \frac{1}{2\pi}\int_{-\pi}^\pi|f(x)|^2dx
     $$

   - 

   - However, it is easy to construct elements in $\ell^2(\Z)$ that do not correspond to functions in $\mathcal{R}$ which is collection of Riemann integrable functions
   - Note that $\ell^2(\Z)$ is complete in its norm, while $\mathcal{R}$ is not.

2) Limits of continuous functions

   - Suppose $\{f_n\}$ is a sequence of continuous functions on $[0,1]$ . We assume that $\lim_{n\rightarrow\infty} f_n(x) = f(x)$ exists for every $x$, and inquire as to the nature of the limiting function $f$.

   - If we did not suppose the convergence is uniform, it is not sufficient to say that 
     $$
     \int_{R^d} f(x) dx = \underset{n\rightarrow \infty}{\lim} \int_{R^d}f_n(x)dx
     $$

3) 

$\rightarrow$ There is a need for a **new integral** that can overcome the limitations of Riemann's integral which is called "Lebegue integral"



### Fundamental theorem of Calculus with Lebegue integral

- What is fundamental theorem of calculus?
  $$
  F(x) = \int _a^xf(y)dy    \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \  (1)\\
  F(b) - F(a) = \int_a^b F'(x)dx\ \ \ \ (2)
  $$

- When $f$ is integrable, we want to see if $F$ is differentiable and above relationship is holds (1)

- We want to find What conditions on $F$ make $F'$ exists and above condition holds (2)



### Preliminaries

#### 	Point

- Let $\R^d$ denote the $d$-dimensional Euclidean space

-  A point $x \in \R^d$ consists of a $d$- tuple of real numbers

  ​          $x = (x_1,x_2,\cdots,x_d),\ x_i \in \R \text{ for } 1\leq i \leq d$

- The norm of a point is denoted by $|x|$ and is defined as the standard euclidean norm

- The distance between two points $x$ and $y$ is $|x-y|$




#### Set

- The complment of a set $E$ in $\R^d$ is denoted by $E^c$

- The reletive complement is denoted by $E-F$ where $E$ and $F$ are subsets of $\R^d$

- The distance between two sets $E$ and $F$ is defined by $d(E,F) = \inf\{|x-y| : x\in E \text{ and } y \in F\}$

- The open ball in $\R^d$ centered at x and of radius r is defined by $B_r(x) = \{y \in \R^d : |x-y| <r\}$

- A set $E \sub \R^d$  is open , if for every $x \in E$, there exists $r>0 $ $s.t\ B_r(x) \sub E$

- A set $E$ is closed if $E^c $ is open

  \* Any union of open sets is open, while the intersection of finitely many open sets is open

  \* Any intersection of closed sets is closed, union of finitely many closed sets is closed.

- A set $E$ is bounded if there is $R>0$ such that $E < B_R(0)$ 

- A bounded set $E $ is complete if it is also closed

  \* Compact sets follow the Heint-Borel covering property

    : Assume $E$ is compact,  $E \in \underset{\alpha \in I}{\bigcup} O_\alpha$ and $O_\alpha$ is open

     Then there are finitely many open sets $O_{\alpha_1},O_{\alpha_2},\cdots,O_{\alpha_n}$ such that $E \in \underset{j = 1}{\overset{n}{\bigcup}}O_{\alpha_j}$

     $\rightarrow$ **Any covering of a compact set contains a finite subcovering!!**



#### 	points of set

- A point $x \in \R^d$ is a limit point of the set $E$ if for every $r>0$, $(B_r(x) - \{x\}) \cap E \neq \empty$ 

  \* A limit point $x$ does not necesarily belong to the set $E$

- The set of all limit points of $E$ is denoted by $E'$

- An isolated point of $E$ is a point $x$ in $E$ such that there is $r>0$ with $B_r(x) \cap E = \{x\}$

- The set of all interior point is called interior of $E$, denoted by $E^o$

- The closure of $E$, denoted by $\bar{E}$, consists of $E \cup \bar{E}$

- The boundary of $E$, denoted by $\partial E$, is the set consist of $\bar{E} - E^o$

  \* A set $E$ is closed $\iff$ $E' \sub E  \iff E  = \bar{E} $

- A closed set $E$ is perfect if $E$ does not have any isolated points.

#### 		

#### 	Rectangle and Cube

- A (closed) rectangle $R$ in $\R^d$ is defined by 
  $$
  R = [a_1,b_1] \times \cdots \times [a_d,b_d], \text{ where }a_j \leq b_j, 1\leq j \leq d
  $$

- Side lengths of $R$ are $b_1-a_1,\cdots,b_d-a_d$

- The volume of $R$ is denoted by $|R|$ and defined by $|R| = (b_1 - a_1) \times \cdots \times (b_d-a_d)$

- A (closed) cube, usually denoted by $Q$, is a rectangle with the same side lengths 

- A union of rectangle is said to be almost disjoint , if the interiors are disjoint



#### Any open sets can be represented by countable union of almost disjoint closed cube

> Lemma 1.1.1 
>
> Let $R, R_j\ j = 1,\cdots,N$ be rectangles such that $R_j$'s are almost disjount and $R = \overset{n}{\underset{j=1}{\bigcup}} R_j$, then $|R|= \sum_{j=1}^n|R_j|$  
>
> Lemma 1.1.2
>
> Let $R, R_j\ j = 1,\cdots,N$ be rectangles s.t $R \sub \bigcup_{j=1}^{n} R_j$, then $|R|  = \sum_{j=1}^{n} |R_j|$

- Theorem 1.1.1

  Every open set $\mathcal{O}$ of $\R$ can be written uniquely as a countable union of disjoint openset.

  > $proof$ 
  >
  > i) For each $x \in \mathcal{O}$ let $I_x$ denote the longest open interval containing $x $ and contained in $\mathcal{O}$ 
  >
  > $I_x = (a_x,b_x)$ where $a_x = \inf \{a<x: (a,x) \sub \mathcal{O}\}$ and $b_x = \sup \{ x<b: (x,b) \sub \mathcal{O}\}$
  >
  > ii) if it can be written by union of disjoint set, there is a rational that can represent each set.  because rationals are countable, The sets are also countable. $\rightarrow$ It is surficient to show that it can be represented by union of disjoint sets
  >
  > iii) When $I_x \cap I_y \neq \empty $, $I_x = I_y$ because of maximality. -> Every open set can be represented as a set of disjoint sets.

  

- Theorem 1.1.2 (multi-dimensional version of thm 1.1.1)

  Every open set $\mathcal{O}$ of $\R^d$, $d \geq 1$, can be written as a countable union of almost disjoint cubes.

  >$proof$
  >
  >i) Define set of cubes$Q_{n_1^k,\cdots,n_d^k}^{\frac{1}{2^{k-1}}}$ that fill an open set $\mathcal{O}$ 
  >
  >ii) because of the definition of $Q$, $\cup Q \sub \mathcal{O}$
  >
  >iii) for any point $x \in \mathcal{O}$, one can find $Q_{n_1^k,\cdots,n_d^k}^{\frac{1}{2^{k-1}}}$ that satisfies $\frac{1}{2^{k-1}} < \var$ . because $x \in B_\var(x)\sub \mathcal{O}$
  >
  >​    $\therefore \mathcal{O} \subset \cup Q$

**$\rightarrow$ The volume of any open sets can be calculated by summations of volumes of cube!!**

