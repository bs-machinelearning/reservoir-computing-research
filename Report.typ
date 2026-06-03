
#set document(
  title: [Reservoir Computing with the \ Human Connectome Project],
)

#set page(
  paper: "a4",
  columns: 2,
  margin: (
    left: 5em,
    right: 5em,
    top: 5em,
    bottom: 5em,
  ),
  numbering: "1",
)

#set heading(numbering: "1.1.")

#place(
  top + center,
  float: true,
  scope: "parent",
)[
  #align(center)[

    #title()
    #grid(
      columns: (1fr, 1fr, 1fr),
      gutter: 1em,
      align: center,
      [
        #link("https://github.com/tebe-nigrelli")[*Tebe Nigrelli*]
      ],
      [
        #link("https://github.com/Kevin-CK3")[*Kevin Cesenj*]
      ],
      [
        #link("https://github.com/christod01")[*Christina E. Christodoulou*]
      ],

      [
        #link("https://github.com/giusiscozzi")[*Giusi Scozzi*]
      ],
      [
        #link("https://github.com/vgkmn1110")[*Vecdi Gokmen*]
      ],
      [
        #link("https://github.com/lorenapavel")[*Lorena Pavel*]
      ],

      [
        #link("https://github.com/paranza")[*Antonio Perazza*]
      ],
      [
        #link("https://github.com/EmilioVitaliEV")[*Emilio Vitali*]
      ],
      [
        #link("https://github.com/a-spahiu")[*Angelo Spahiu*]
      ],
    )

    Following the kind direction of Postdoc Researcher #link("https://scholar.google.com/citations?user=MykmXMQAAAAJ")[Victor Buendìa]

    #line(length: 100%)
  ]
]

#v(0.5em)
#align(center)[*Abstract*]

This project studies reservoir computing with human structural connectomes. We validate a standard pipeline on the Lorenz system, then replace the random recurrent matrix with a weighted Human Connectome Project adjacency matrix while training only a ridge-regression readout. Gaussian, connectome, leaky, and perturbed reservoirs are compared. All models recover bounded Lorenz-like dynamics but diverge quickly in exact trajectory prediction, consistent with chaos. Perturbations show node deletion is most damaging, while edge deletion and rewiring are weaker because much of the recurrent capacity and spectral scaling remains intact.

#v(1em)

= Introduction

The brain processes information through networks of neurons connected by synapses. Reservoir computing follows a related principle: a fixed recurrent network, the *reservoir*, transforms an input signal into a rich set of internal states, from which a simple linear readout is trained to make predictions. Since only the readout is trained, the method remains efficient while still modeling complex and chaotic signals.

Here we use structural connectivity data from the Human Connectome Project, which maps connections between brain regions. This lets us replace the usual random reservoir with a biologically grounded network that embeds brain structure directly in the model.

The reservoir is driven by a chaotic input signal and trained to predict the next time step. We then remove nodes and connections to simulate structural damage and measure how prediction degrades. This tests how connectome topology supports computation and how robust it is to perturbations.



= Reservoir dynamics

A reservoir can be understood through the analogy of a lake. Raindrops create ripples that spread, interact, and persist before fading; by observing the surface, one can infer something about the rain without seeing each drop. Similarly, an input signal perturbs the network, and that perturbation propagates through the connections and shapes later states.

This intuition can be formalized by describing the reservoir as a dynamical system. The state of neuron $i$ at time $t$, denoted by $r_i(t)$, evolves as

$
  (d r_i) / (d t) = -r_i (t) + S(sum_(j=1)^N W_(i j) r_j (t))
$

The term $-r_i(t)$ represents decay, while the second term captures recurrent input through the weight matrix $W$ after a nonlinear transformation. The current reservoir state is therefore a compressed representation of recent inputs.

In practice, its dynamics are approximated by evaluating the state at discrete time steps. The time derivative is expressed as

$
  frac(d r_i (t), d t) = frac(r_i (t + delta t) - r_i (t), delta t)
$

which leads, via Euler discretization, to the update rule

$
  r_i (t + delta t) = r_i (t) + delta t (d r_i) / (d t)
$
The richness of reservoir dynamics comes directly from its structure: an irregular, highly connected network turns even simple inputs into complex activity patterns. Because neurons respond differently, the network produces a diverse feature set from which the readout can extract information.

This balance is governed by the recurrent matrix $W$, which determines whether activity decays, persists, or grows. The resulting dynamical regime plays a central role in the reservoir's computational capacity.

To study this behavior in a controlled setting, we first use a Gaussian random reservoir, where the recurrent matrix $W$ is built from independent Gaussian entries sampled as

$
  W_(i j) ~ cal(N)(0, frac(J^2, N))
$

where $N$ is the number of neurons and $J$ controls the overall strength of recurrent interactions.

System stability can be understood through the eigenvalues of $W$. For large $N$, Girko's circular law places them approximately within a disc of radius $J$ in the complex plane. The spectral radius, the magnitude of the largest eigenvalue, therefore determines whether perturbations decay or grow. In discrete time, stability requires eigenvalues inside the unit circle; in continuous time, it depends on their real parts.

Thus, three regimes appear: for $J < 1$, activity decays; for $J approx 1$, the system is near criticality; and for $J > 1$, the dynamics become unstable or chaotic.

External input enters through the matrix $W^"in"$. In the state equation, it appears as $sum_j W^"in"_(i j) h_j(t)$ and determines how the input affects each neuron before the nonlinearity. This fixed, randomly initialized matrix distributes the input across the reservoir, so the state reflects both internal dynamics and incoming signals.

The goal is to use the reservoir state to predict the next input value, $h(t) approx h(t + delta t)$. The model maps $r(t)$ to the output through a linear transformation:
$h(t) = W_"readout" * r(t)$, where $W_"readout"$ contains the readout weights.

The internal and input weights remain fixed; only the readout weights are adjusted during training to approximate the next-step signal, $h(t) approx h(t + delta t)$.

The reservoir's internal structure is therefore not trained. Its computational power comes from fixed connections and their induced dynamics rather than from adapting recurrent weights during learning.

#figure(
  image("img/time_predictor.png", width: 100%),
  caption: [Reservoir time-series prediction example.],
)

= Dynamics overview


Building on the framework introduced above, we now investigate how the reservoir dynamics change as the coupling parameter $J$ is varied.

Figure 2 shows representative neuron trajectories for different values of $J$. For $J < 1$, trajectories quickly converge to zero, so the system loses dependence on its initial state. Near the critical value, activity decays more slowly and retains information longer. For $J > 1$, trajectories become irregular and highly variable, indicating a transition to chaos.

#place(
  top + center,
  float: true,
  scope: "parent",
)[
  #figure(
    grid(
      columns: 3,
      gutter: 1em,

      image("img/figure1.png", width: 100%),
      image("img/figure1.1.png", width: 100%),
      image("img/figure1.2.png", width: 100%),
    ),
    caption: [Neuron trajectories for $J < 1$, $J approx 1$, and $J > 1$.],
  )
]

To quantify this transition, we measure temporal variability. Networks of $N = 400$ neurons are simulated for $T = 50$ time units with step size $d t = 0.1$. For each $J$, the variance of each neuron's activity is computed over time and averaged across neurons and $10$ fixed-seed realizations.

The results are shown in Fig. 3. For small $J$, the mean temporal variance stays near zero, confirming rapid convergence. Beyond the critical regime, the variance rises sharply, reflecting sustained fluctuations.

#figure(
  image("img/figure2.png", width: 100%),
  caption: [Mean temporal variance vs. $J$ (10 trials, seed 42).],
)

These regimes directly affect reservoir memory. In the stable regime ($J < 1$), the system quickly forgets past inputs as trajectories collapse to a fixed point. Near criticality ($J approx 1$), slower decay improves memory. For $J > 1$, sensitivity to perturbations breaks consistent behavior and violates the Echo State Property.

The same transition appears at the population level. In Fig. 4, activation distributions broaden as $J$ increases, shifting from a narrow concentration around zero to a wide, heterogeneous spread.

#figure(
  image("img/figure3.png", width: 100%),
  caption: [Activation histograms over time ($N=500$, $T=100$, $d t=0.1$).],
)

= Training process

Before presenting the results, we outline the implementation strategy. The project has two phases. First, we built and validated the predictive pipeline with standard Gaussian random networks. After confirming that the baseline could model the Lorenz attractor in a controlled setting, we replaced the random matrices with empirical human connectome data to test the same task on a brain-derived topology.

In reservoir computing, the internal network is fixed, so training has two stages: simulate the reservoir to collect internal states, then tune the regularization to optimize the readout weights.

To generate data for readout optimization, we simulate the reservoir's response to the target signal using *teacher forcing*: the reservoir is driven by the true Lorenz signal. At each time step, all neuron activations are stored in a state matrix $R$.

During this phase, we discard a washout period of $200$ time steps. This removes the transient caused by zero-state initialization, so $R$ reflects synchronized Lorenz-driven dynamics.

Figure 5 shows the post-washout input signal and the activations of $8$ sampled neurons. For this Gaussian baseline, we used $N = 250$ neurons, spectral radius $J = 0.8$, and input scaling $0.1$. The activations are heterogeneous without saturating, giving the regression model a useful high-dimensional representation.

Once $R$ is collected, the output weights $W_"readout"$ are computed with ridge regression. To learn forward dynamics, the reservoir state at time $t$ predicts the target at $t+1$; the shifted targets form $H_"target"$.

The weights are then obtained by solving a regularized least-squares problem:

$
  W_"readout" = arg min_W |W R - H_"target"|^2 + lambda |W|^2
$

where $lambda$ penalizes large weights and helps prevent overfitting to noise or small fluctuations in the training data.

We also augment the state matrix with a constant bias term to capture spatial offsets without distorting the weights.

The main experimental focus in this phase was tuning $lambda$. On the baseline network ($N = 250$), we tested $30$ logarithmically spaced values from $10^{-18}$ to $1$.

#figure(
  image("img/figure5.png", width: 80%),
)

Increasing the regularization penalty raises training RMSE across all three spatial axes. The best training fit occurs for very small $lambda$ values, around $10^{-8}$.

= Human connectome data and network analysis

We now turn to the human connectome data and examine how its graph structure can serve as a biologically grounded reservoir.

The dataset comes from a preprocessed Human Connectome Project release in which each subject is stored as a `.graphml` file representing brain structural connectivity.

Each file records brain regions and the connections between them. Nodes correspond to regions, while edges represent structural connections such as white matter fiber tracts.

Graph-theoretically, the connectome is modeled as $G = (V, E)$, where $V$ is the node set and $E$ the edge set. The graph is undirected, so the connectivity matrix is symmetric: $W_(i j) = W_(j i)$.

The analyzed graph has $463$ nodes, determined by a brain parcellation that balances anatomical detail with computational cost. This resolution remains manageable while preserving meaningful connectivity structure.

A relatively large node count also makes network statistics clearer. In smaller graphs, finite-size fluctuations can obscure the underlying structure.

Each node includes its 3D position, hemisphere, and anatomical label. Edges describe regional connections and include mean fiber length, fractional anisotropy (FA), and number of fibers.

We separate the graph into node- and edge-level information to analyze both spatial organization and connection structure.

At the node level, positions show a clear 3D brain organization. The distribution is roughly symmetric, with visible separation between hemispheres; most regions form a compact structure, while boundary nodes are more spread out. This confirms that the graph preserves anatomical layout.

#figure(
  image("img/figure6.png", width: 100%),
  caption: [Connectome nodes in 3D, colored by hemisphere.],
)

At the edge level, connection features are strongly heterogeneous. The number of fibers ranges from about $1$ to more than $4600$, with median around $22$, so most connections are weak. The distribution is strongly right-skewed, with many low values and a long tail of high-weight edges.

The high skewness ($approx 6.37$) and large kurtosis ($approx 63$) support this heavy-tailed interpretation.

Fiber length spans a wider range, roughly $10$ to $100$, and decays more smoothly, with moderate skewness.

The FA distribution is more concentrated and decreases rapidly as values increase. Visually it resembles an exponential-like decay, though a formal test would be needed to confirm this.

Overall, the connectome is heterogeneous rather than uniform, with a small subset of connections carrying much larger weights.

#figure(
  image("img/figure 7.png", width: 100%),
  caption: [Edge-feature histograms and pairwise densities.],
)

We further examine the edge features through correlation analysis. Most relationships are weak. The clearest pattern is a moderate negative correlation between fiber length and number of fibers ($rho approx -0.37$), suggesting that longer connections tend to involve fewer fibers.

Correlations involving FA are much weaker ($rho approx 0.08$ with fiber length and $rho approx 0.04$ with number of fibers), suggesting that FA captures a different aspect of connectivity.

We use Spearman's rank correlation rather than Pearson correlation because the edge features, especially fiber count, are strongly skewed. Rank correlation is less dominated by extreme values and is therefore more robust here.

#figure(
  image("img/figure8.png", width: 100%),
  caption: [Spearman correlations between structural edge features.],
)

Finally, the network shows non-trivial *community structure*: groups of nodes are more strongly connected internally than with the rest of the graph. This supports the view of the connectome as a structured, not random, network.

== Training and prediction on the Lorenz system

We next introduce the chaotic system used to test the reservoir, chosen for its simplicity and rich nonlinear dynamics.

The Lorenz system is a set of three coupled ordinary differential equations developed by Edward Lorenz while studying atmospheric convection. It is a classic example of deterministic chaos: the equations contain no randomness, yet they are highly sensitive to initial conditions and can display aperiodic behavior.

$
  cases(
    (d x) / (d t) = sigma (y - x),
    (d y) / (d t) = x (rho - z) - y,
    (d z) / (d t) = x y - beta z
  )
$

The parameters $rho$, $sigma$, and $beta$ are constants, while $x(t)$, $y(t)$, and $z(t)$ are time-dependent state variables.

Sensitivity to initial conditions makes long-term prediction infeasible. Although the system is deterministic, small numerical errors are amplified exponentially, so trajectories eventually diverge from the true continuation.

Another key property is boundedness. Although trajectories are chaotic and sensitive, they do not diverge to infinity; instead, they remain confined to the butterfly-shaped Lorenz attractor.

This makes the system useful for reservoir computing: beyond short-term accuracy, it tests whether the model learns the qualitative attractor geometry. A good autonomous model should remain bounded even after exact phase alignment is lost.

We generate the Lorenz attractor by numerical integration with the Euler method, chosen for speed. The classical Lorenz-63 parameters place the system in a chaotic but bounded regime that produces the familiar butterfly attractor.


= Results

The final experiments, collected in `4. Reference/`, all test the same task: train a fixed reservoir on teacher-forced Lorenz dynamics, then run it autonomously and compare the prediction with the true continuation. Qualitatively, we ask whether the model remains on a bounded Lorenz-like attractor. Quantitatively, the perturbation notebooks use divergence time $t_"div"$, the first time at which the Euclidean prediction error exceeds a threshold $epsilon$.

== Gaussian random reservoir baseline

The Gaussian baseline in `gaussian_reservoir_implementation.ipynb` uses a dense random recurrent matrix rather than a connectome. In the recorded run, the reservoir has $N = 300$, target spectral radius $J = 0.85$, input scaling $0.5$, leak parameter $alpha = 1$, and a training window of $10000$ time units. The actual spectral radius is $0.8500$, and the full workflow takes about $26.4$ seconds.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,

    image("img/gaussian_components.png", width: 100%), image("img/gaussian_attractor.png", width: 100%),
  ),
  caption: [Gaussian reservoir prediction: components and 3D attractor.],
)

Qualitatively, the Gaussian reservoir produces a bounded oscillatory trajectory, but it soon loses phase alignment with the true Lorenz signal. In 3D, the prediction visits the same broad phase-space region but traces loops that are too regular and too spread out.

This suggests that the baseline learned a coarse autonomous oscillator rather than the exact Lorenz flow. A likely reason is the shift from one-step training to closed-loop evaluation: small one-step errors are fed back and amplified by chaotic dynamics. The dense topology provides many features, but these hyperparameters do not yield stable long-horizon tracking.

== Human-connectome reservoir

The main connectome experiment in `main.ipynb` replaces the Gaussian recurrent matrix with the weighted adjacency matrix of a human connectome graph. The notebook uses $J = 0.9$, input scaling $0.5$, ridge parameter $lambda = 3e-6$, and a training horizon of $20000$ time units. The workflow takes about $111.0$ seconds, with reported divergence time $t_"div" approx 0.39s$ for $epsilon = 1.0$.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,

    image("img/connectome_components.png", width: 100%), image("img/connectome_attractor.png", width: 100%),
  ),
  caption: [Connectome reservoir prediction: components and 3D attractor.],
)

The component plot shows correct initial scale and oscillatory structure, followed by rapid loss of alignment. In 3D, the prediction remains bounded and Lorenz-like but expands away from the true attractor.

The short divergence time is expected because the Lorenz system amplifies small errors exponentially. More importantly, the connectome reservoir captures some qualitative geometry even though its internal weights are anatomical rather than optimized for this task. The mismatch reflects that a biologically motivated topology need not be ideal for this particular chaotic signal.

== Leak-rate connectome reservoir

The leaky version in `main_with_leak_rate.ipynb` changes the state update to mix the old state with the new nonlinear activation. The recorded run uses $alpha = 0.3$, $J = 0.9$, input scaling $0.5$, $lambda = 3e-6$, and a shorter training horizon of $1000$ time units. The notebook reports a total runtime of about $3.74$ seconds.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,

    image("img/leaky_connectome_components.png", width: 100%), image("img/leaky_connectome_attractor.png", width: 100%),
  ),
  caption: [Leaky connectome prediction ($alpha = 0.3$).],
)

The leaky reservoir produces smoother autonomous dynamics. The prediction follows the initial state more gently before drifting, and the 3D attractor remains bounded. However, it still deviates from the true trajectory and does not reproduce the exact lobe-to-lobe switching pattern.

The leak rate adds inertia to the reservoir state. This can preserve memory and prevent abrupt updates, but it also slows the response to new input. The result is a trade-off: dynamics are smoother, but exact chaotic prediction still fails quickly. Because this run uses a shorter training window, it should be read as a qualitative comparison rather than a fully tuned improvement.

== Perturbation experiments

The perturbation experiments in `run_experiment.ipynb` evaluate robustness under four transformations: node deletion, edge deletion, edge swapping, and node-label swapping. For each perturbation level $p$, divergence time is averaged across multiple connectome files and random seeds.

#figure(
  image("img/perturbations_all_modes.png", width: 100%),
  caption: [Divergence time under four connectome perturbations.],
)

Node deletion has the strongest effect. As $p$ increases, average divergence time drops sharply and approaches zero when most nodes are removed. This is expected: removing nodes reduces reservoir dimension, leaving the readout fewer variables with which to reconstruct the Lorenz state.

Edge deletion is less damaging over most of the range and drops strongly only near extreme deletion. This suggests substantial redundancy: many edges can be removed without immediately destroying short-term dynamics. A methodological factor also matters: in `experiment_functions.py`, each perturbed graph is rescaled by its own spectral radius, keeping recurrent gain comparable and partly masking edge-deletion effects.

Edge swapping has little visible effect in the plotted range. For this task and hyperparameter setting, preserving node count and global recurrent strength matters more than preserving exact anatomical edge placement. Node swapping also has little effect because it mostly changes labels or ordering while leaving the graph nearly isomorphic.

== High-damage edge-deletion sweep

The saved edge-deletion figure and `results.npz` file give a closer view of high perturbation levels. The mean divergence time changes only modestly from $p = 0$ to $p = 0.90$, then decreases more clearly at $p = 0.95$ and $p = 0.99$. In the saved arrays, the mean is about $0.51s$ at $p = 0$ and $0.36s$ at $p = 0.99$.

#figure(
  image("img/drop_edges_high_damage.png", width: 100%),
  caption: [High-damage edge-deletion sweep.],
)

This supports the interpretation that the connectome reservoir is robust to moderate edge loss. Small or moderate deletion may remove mostly redundant connections, while the remaining weighted graph still supports short-horizon prediction. At very high deletion, the graph loses too much connectivity to maintain useful autonomous dynamics.

= Conclusion

The experiments show that both Gaussian and connectome-based reservoirs can generate bounded Lorenz-like trajectories, but neither gives reliable long-horizon prediction under the tested settings. This is consistent with the Lorenz system: once the model runs autonomously, small readout errors are recursively amplified. The connectome result remains meaningful because it shows that a fixed anatomical graph can act as a reservoir with nontrivial computational dynamics.

The perturbation results are the clearest finding. Removing nodes is far more damaging than removing or rewiring edges because node deletion directly reduces reservoir dimensionality. Edge deletion and swapping have weaker effects, likely due to redundant pathways and spectral rescaling that preserves global recurrent strength. The leak-rate experiment shows that slower updates can smooth predictions, but not solve divergence.

Future work should tune $J$, $alpha$, input sparsity, and regularization systematically, evaluate multiple connectome subjects with consistent data paths, and compare perturbations with and without spectral rescaling. This would separate topology from global gain and strengthen the biological interpretation of robustness.
