<div style = "text-align : center"><font size = "5em"> Exterior measure</div>

- Definition 1.2.1

  Let $E$ be any subset of $\R^d$. The exterior measure( = outer measure) of $E$ is defined by 
  $$
  m_*(E) = \inf \displaystyle \sum_{j=1}^\infty |Q_j| 
  $$
  where the infimum is taken over all countable covering $E \sub \bigcup_{j \in \N} Q_j$ by closed cube

  

- Remark 1.2.1

  1) It is not sufficient to allow finite sums in the def of $m_*(E)$
  2) The covering by cubes in the def of $m_*(E)$ can be replaced by coverings of rectangles or balls
  3) The exterior measure of a point is zero
  4) The exterior measure of a closed cube is equal to its volume $m_*(Q) = |Q|$
  5) The exterior measure of a rectangle is equal to is volume
  6) The exterior measure of $\R^d$ is infinite

  $\rightarrow$ by rmk 3. and 6. $m_*$ is a mapping function $\{E \sub \R^d\} \rightarrow [0,\infty]$



- Theorem 1.2.1 (properties of exterior measure)

  1) for every $\epsilon>0$,  there exists a covering $E \sub \bigcup_{j\in\N} Q_j$ s.t $\sum_{j}|Q_j| \leq m_*(E) + \epsilon$  (by def of infimum)

  2) (Monotonicity) if $E_1 \sub E_2$, then $m_*(E_1) \leq m_*(E_2)$

     > $pf$
     >
     > i) Let $\{Q_{ij}\}$ be a coverings of $E_i$. $E_i\sub \{Q_{ij}\}$
     >
     > ii) $E_1 \sub E_2 \rightarrow $ $\{Q_{2j}\} \sub \{Q_{1j}\}$ 
     >
     > iii) $m_*(E_1) \leq m_*(E_2)$

  3) (countable sub-additivity) if $E \sub \bigcup_{j=1}^{N}E_j$, then $m_*(E_1) \leq m_*(E_2)$ 

     > $pf$
     >
     > i) When $m_*(E_j) = \infty$ for some $j$, the inequality holds
     >
     > ii) Assume that $m_*(E_j) < \infty, j \in \N$ 
     >
     > iii) $\sum_k|Q_{j,k}| \leq m_*(E_j)+ \frac{\epsilon}{2^j}$ for covering $\{Q_{j,k}\}$ of $E_j$ $(\because \text{Thm 1.2.1-1})$
     >
     > iv) $E \sub \cup_jE_j \sub \cup_j \cup_k Q_{j,k}$
     >
     > v) $m_*(E) \leq \sum_j\sum_k|Q_{j,k}|  \leq \sum_j(m_*(E_j) + \frac{\epsilon}{2^j}) = \sum_jm_*(E_j) + \epsilon$

  4) Let $E \sub \R^d$, $m_*(E) = \inf m_*(\mathcal{O})$ where the infimum is taken over all open sets $\mathcal{O} \supset E$

     >$pf$
     >
     >i) $m_*(E) \leq m_*(\mathcal{O})$ by monotonicity
     >
     >ii) We can choose closed covering $\sum_j|Q_j| \leq m_*(E) + \frac{\epsilon}{2}$ by def of exterior measure
     >
     >iii) we can choose open cube $|Q_j^o| \leq |Q_j| + \frac{\epsilon}{2^{j+1}}$  $\because$ volume of open set and closed set is same
     >
     >iv) Set $\mathcal{O} = \bigcup_jQ_j^o$ then $m_*(\mathcal{O}) \leq \sum_j|Q_j^o| \leq \sum_j|Q_j| + \frac{\epsilon}{2} \leq m_*(E) + \epsilon$

  5) If $E = E_1 \cup E_2$, and $d(E_1, E_2) >0$, then $m_*(E) =m_*(E_1) + m_*(E_2)$

     > i) Let $d(E_1,E_2) = \delta >0$
     >
     > ii) Then we can find covers of  each $E_j$ that has diameter less than $\var$ -> It can be divided into which of the two sets are included. We can index it by $J_1\text{ and }J_2$
     >
     > iii) $m_*(E_1) + m_*(E_2) \leq \sum_{J_1}|Q_j| + \sum_{J_2}|Q_j| = \sum_j|Q_j| \leq m_*(E) + \epsilon$ ($\geq$)
     >
     > iv) $m_*(E) \leq m_*(E_1) + m_*(E_2)$ by sub-additivity ($\leq$)

  6) If a set $E$ is countable union of almost disjoint cubes $E = \bigcup Q_j$, then $m_*(E) = \sum|Q_j|$

     > i) ($\leq$) holds because of sub-additivity
     >
     > ii) Define $|Q_j| \leq |\tilde{Q_j}|+ \frac{\epsilon}{2_j}$ (a smaller cube contained in $Q_j$). and mutually disjoint
     >
     > iii) $m_*(E) \geq m_*(\bigcup_j\tilde{Q_j}) = \sum_j|\tilde{Q_j}| \geq \sum_jQ_j - \epsilon$ ($\because item(5)$ )  $\rightarrow$ $(\geq)$ holds

  

 