# Asset-Allocation

## Introduction:

Modern portfolio management requires solving a fundamental challenge: how should an investor distribute capital across many assets in order to maximize returns while minimizing risk? This question, first formalized by Harry Markowitz in 1952, forms the basis of what is known as the portfolio optimization problem. The classical Markowitz approach seeks to minimize the variance of a portfolio subject to constraints on expected return, but solving this problem exactly becomes computationally expensive as the number of assets grows, since general quadratic optimization problems (QUBOs) are NP-hard. As financial markets grow in complexity and the number of tradable assets increases, there is a growing need for more scalable approaches to portfolio optimization.

Quantum annealing, implemented by hardware such as D-Wave's quantum processors, solves optimization problems by encoding them as Quadratic Unconstrained Binary Optimization (QUBO) problems. The goal is mainly to minimize a function of binary variables with both linear and quadratic terms. Prior research has shown that quantum annealers can produce portfolios achieving more than 80\% of the return of classical optimal solutions while satisfying risk constraints, suggesting real promise for quantum approaches in finance. Separately, researchers have demonstrated that quantum annealers can effectively detect community structure in networks by maximizing a modularity metric, a technique that identifies groups of densely connected nodes within a graph.

This project combines both of these ideas into a two-level hierarchical portfolio allocation framework. Using historical return data from 20 publicly traded stocks downloaded via Yahoo Finance, we first construct a covariance graph representing how asset returns move together. We then apply a quantum annealing-based community detection algorithm — formulated using the modularity matrix — to partition the 20 stocks into four communities of correlated assets. Within each community, we apply a second QUBO-based optimization to determine individual asset allocations. Our research investigates whether this hierarchical quantum approach can produce meaningful portfolio allocations, and examines practical challenges that arise when encoding financial constraints, such as the normalization requirement, into QUBO formulations.

<img width="723" height="623" alt="Screenshot 2026-04-28 at 3 06 36 PM" src="https://github.com/user-attachments/assets/5b74a187-38ff-46fd-a12d-75881bf23afc" />

## Set-Up:

First pull the repo in some directory you are conformatble working in.

``` 
git clone https://github.com/dhruvupreti05/Asset-Allocation.git
```

Note that the setup file works only for Unix machines, windowsOS users must create a virtual environment and install the libraries in `requirements.txt`. For Unix machines change the permissions of the setup file via `chmod +x setup.sh` first then run the setup, `./setup.sh` which will create a virtual environment and install the packages. After that, whenever you want to use the project, activate the virtual environment by running `source ocean/bin/activate` to activate the virtual environment.

## References:

- Xu, H., Dasgupta, S., Pothen, A., & Banerjee, A. (2023). [Dynamic asset allocation with expected shortfall via quantum annealing](https://arxiv.org/pdf/2112.03188). Entropy, 25(3), 541.
- Negre, C. F., Ushijima-Mwesigwa, H., & Mniszewski, S. M. (2020). [Detecting multiple communities using quantum annealing on the D-Wave system](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0227538&type=printable). Plos one, 15(2), e0227538.
- Cadavid, A. G., Kaushik, A., Chandarana, P., Lopez-Ruiz, M. A., Dev, G., Aboumrad, W., ... & Hegade, N. N. (2026). [Large-scale portfolio optimization on a trapped-ion quantum computer](https://arxiv.org/pdf/2602.23976). arXiv preprint arXiv:2602.23976.

This work was created by the collaborative efforts of Bhavya Lakhina, Esha Sury, Dhruv Upreti and Shashank Boopathi; mentorship was given by Kale Stahl and Birgit Kaufmann. Funding was given through D-wave. 

