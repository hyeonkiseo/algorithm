<div style = 'text-align : center'><font size = '5em'>Measurable set
</div>

#### def 1.3.1 (lesbegue measure)

A set $E$ of $\R^d$ is (Lesbegue) **measurable** if for any $\epsilon >0$ , there exists an open set $\mathcal{O}$ such that $E \sub \mathcal{O}$ and $m_*(\mathcal{O}-E) \leq \epsilon$. If $E$ is measurable, we can define its **(lesbegue) measure** $m_(E)$ by $m(E) = m_*(E)$

 

#### Theorem 1.3.1 (properties of measurable set)

1) Every open set in $\R^d$ is measurable
2) If $m_*(E) =0$, then $E$ is measurable. In particular, if $F \sub E$ where $m_*(E) =0$, then $F$ is measurable.
3) A countable union of measurable sets is measurable
4) Closed sets are measurable.
5) The complement of a measurable set is measurable  ($\rightarrow$ **the collection of measurable sets are $\sigma-$algebra**)
6) A countable intersection of measurable sets is measurable.



#### Lemma 1.3.1 (distance between closed set and compact set)

If $F$ is closed and $K$ is compact and those are disjoint, then $d(F,K) > 0$.

(**distance between disjoint closed set $\ngtr0$ in general**)



#### Lemma 1.3.2 (measure of relative complement)

Let $E_1$ and $E_2$ be measurable sets such that $E_1 \sub E_2$.  If $m(E_1)<\infty$ , then $m(E_2 - E_1) = m(E_2) - m(E_1)$ 



#### Theorem 1.3.2 (countable additivity)

If $\{E_j\}_{j\in\N}$ is a sequence of mutually disjoint measurable sets, and $E = \bigcup_{j\in\N}E_j$. Then $m(E) = \sum_{j\in\N}E_j$



#### Corollary 1.3.1 (continuity from above & continuity from below)

Suppose $\{E_j\}_{j\in\N}$ is a sequence of measurable sets and $E = \bigcup_{j\in]N}E_j, F = \bigcap_{j\in\N}E_j$

1. If $E_k \sub E_{K+1}, \ k\in\N$, then $m(E) = \underset{j\rightarrow\infty}{\lim}m(E_j)$
2. If $E_{k+1}\sub E_k, k \in \N\text{ and } m(E_1) < \infty$, then $m(F) = \underset{j\rightarrow\infty}{\lim}m(E_j)$   ($ m(E_1) < \infty$condition is essential. think of $E_j = (j,\infty)$ case.)



#### Thm 1.3.3 (well-approximation for measurable sets)

Suppose $E$ is measurable, then for every $\epsilon >0$,

1. There exists an **open set $O$** such that $E \sub O$ and $m(O - E) \leq \epsilon$
2. There exists a **closed set $F$** such that $F \sub E \text{ and }m(E-F) \leq \epsilon$
3. If $m(E) < \infty$, then there exists a **compact set $K$** with $K \sub E$ such that $m(E - K) < \epsilon$
4. If $m(E) < \infty$, there exists a finite union $F  = \bigcup_{j=1}^NQ_j$ of closed cubes such that $m(E\triangle F) \leq \epsilon$.

$\rightarrow$ **Sets satisfying above condition in exterior measure are measurable!!**



#### Invariance properties 

- Translation invariance 

  For a measurable set $E, h\in\R^d$, define $E_h = E + h = \{x+h : x \in E\}$. Then $E_h$is also measurable and $m(E_h) = m(E)$

- Dilation invariance 

  For a measurable set $E,\delta>0$, define $\delta E = \{\delta x : x \in E\}$. Then $\delta E$is also measurable and $m(\delta E) = \delta^d m(E)$

- Reflection invariance 

  For a measurable set $E$, define $-E = \{- x : x \in E\}$. Then $- E$ is also measurable and $m(- E) = m(E)$



#### Def 1.3.2 $\sigma$- algebra and Borel Sets

1. A **$\sigma$-algebra of sets** is a collection of subsets of $\R^d$ that is closed under countable union, countable intersection and complements.

2.  The **Borel $\sigma$-algebra ** in $\R^d$, denoted by $\mathcal{B}_{\R^d}$, is the smallest $\sigma$-algebra that contains all open sets. An element of the Borel $\sigma$-algebra is called a **Borel set** 

3. **$G_\delta$ sets** are countable intersections of open sets and **$F_\sigma$ sets** are countable unions of closed sets. 

   (infinite intersection of open sets and infinite union of closed set are not closed in open and closed respectively. That is why these concepts are defined. They are also in Borel $\sigma$-algebra)



#### Thm 1.3.4 (Equivalence of Lesbegue measure)

Let $E$ be a subset of $\R^d$. Then the followings are equivalent.

1. $E $ is measurable

2. For every $\epsilon>0$, There exists an open set $O$ with $E \sub O$ such that $m_*(O-E) \leq \epsilon$

3. For every $\epsilon > 0$, There exists a closed set $F$ with $F \sub E$ such that $m_*(E - F) \leq \epsilon$

4. There exists a $G_\delta$ set $\tilde{G}$ with $E \sub \tilde{G}$ such that $m_*(\tilde{G}-E) = 0$

5. There exists a $F_\sigma$ set $\tilde{F}$ with $\tilde{F}\sub E$ such that $m_*(E - \tilde{F})=0$

6. For each set $A \sub \R^d$, $m_*(A) = m_*(A\cap E) + m_*(A\cap E^c)$

   Suppose further that $m_*(E) < \infty$

7. For every $\epsilon >0$, there exists a finite union $F = \bigcup_{j=1}^NQ_j$ of closed cubes such that $m_*(E\triangle F) \leq \epsilon$

