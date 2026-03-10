== 2026-03-04, Buendìa + Full Group Meeting
He will give us steps to follow, so each piece is easy to check using simple artificial models. As a last step, you insert the real-world data and you can be sure that the results are as expected.
Reservoir Computing - Echo State Network

== Components
Input: we can vary what data we put in. It has a structure, but we pick something as hard as possible, so it's significant that the model can model it. We will use the Lorenz system (chaotic system - butterfly effect).
Reservoir: you don't train it, just let it run. We have different patients, also at different levels of precision - which means you can compare results in time.
Output: train a linear model to predict x(t + dt) from x(t). You can evaluate performance as MSE for the difference between true value and predicted value after some time. You could also be interested in something weaker, which is the histogram.

It's like having a lake, with raindrop, and from the waves trying to understand where the rain will land next.

The brain is modelled as a network of nodes (neurons), where each neuron has a rate function r_i(t) which models the number of spikes that each neuron “i” experiences at a moment of time.

The differential equation model describes a function “locally” by expressing its derivative as a function. Solving for the function is hard, but you can easily simulate it with a computer: since the derivative can be approximated as a finite difference (approximate its tangent), you can use Euler's Method to model it.

Hyperbolic Tangent activation.

There will be a moment to experiment with many ideas. The question is which features of the network affect the performance of the model.

The dataset has a lot of data and it is public. It is also quite popular and quite large, meaning it has a lot of uses. Almost all data is from healthy people. It also has some raw data, but then data preprocessing gets hard and you need to do some advanced “cleaning” for it. Human Connectome Project

We are interested in variability between people, and between areas of the brain in a cluster way, since there's a lot of fractal / self-similarity..

We will start with the network:

Build the dynamical system with an input whose consequences on the network are easy to predict. Use Numpy so the operations are vectorized. Be careful with cache invalidation. A common network is the Gaussian Network, $W_(i j) ~ N(0, J/sqrt(N))$, because if you are using a hyperbolic tangent for activation, since J < 1 results in convergence to zero of the rates, while J > 1 results in chaotic behavior. We will run experiments with multiple networks, sampling the connectivity matrix each time. N=300 should begin to show results. 
