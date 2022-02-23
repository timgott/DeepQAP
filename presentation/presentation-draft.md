% Quadratic Assignment Problem with Reinforcement Learning

---

# The quadratic assignment problem

[Image of 2 graphs]

[Demonstrate assignment and show how to compute value]

[mention traditional algorithms and scaling problems to motivate machine learning]

---

# Mathematical formulation [?]

...

---

# Graph Neural Networks

[Image showing weighted graph
-> Icon for messages on edges (message transformation)
-> Arrows from messages to nodes (aggregation)
]

$$
\mathbf{x}_i' = \psi \left( \mathbf{x}_i, \sum_{j \in \mathcal{N}(i)} \phi(\mathbf{x}_j, \mathbf{e}_{ij}) \right)
$$

---

# Reinforcement learning environment

State
: Input graphs

Actions
: Pairs of nodes

Reward
: (Negative) cost that becomes fixed after pair has been assigned

Next state
: Graph after pair has been assigned

---

# Q Network

Predicts the best achievable value after taking an action

[image]

1. Encode graph structure into node embeddings with two separate GNNs
2. For every pair of nodes, compute predicted value after assignment

---

# Representing previously assigned nodes in graph

Options:

- Add a binary feature to the node
- Use special network to encode nodes
- Compute an equivalent subproblem for the remaining nodes

---

# Subproblems after assignment

[Image to illustrate the approach]

Requires heterogenous graph neural network

---

# Limitations of Graph Neural Networks

[Show triangle and rectangle graph that a GNN cannot distinguish]

Isomorphism can be represented as QAP, but GNN cannot solve it

Oversmoothing of node embeddings

---

# Remedies for GNN limitations

- More expressive GNNs (incorporating higher-order structures, initializing node features to be distinguishable)
    - Computationally expensive, harder to train
- Normalization layers to force distance between node embeddings
    - Possibly strong distortion with few nodes
    - Eliminate global information contained in nodes
- Local search

---

# Evaluation

[reward plot on single instance]

[reward plot on random problem distribution]

[q value compared to stochastic/optimal q values]

---

# Open questions

- What is the impact of state representation on performance?
- Which patterns does the GNN need to be able to recognize? How to choose training data?
- What is the heuristic learned by the agent?
- How import is exploration in this task?

---

# Not enough time to talk about

- QAP normalization (scale and shift invariance) / diagonal elimination
- REINFORCE agent
- Details about experiments and evaluation
- Edge weight histogram idea
- Directed graphs
- Details and impact of normalization layers
