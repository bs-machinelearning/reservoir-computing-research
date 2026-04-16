
#set document(
  title: [Reservoir Computing with \ the Human Connectome Project],
)

#set page(
  paper: "a4",
  columns: 2,
  margin: (
    left: 5em,
    right: 5em,
    top: 4em,
    bottom: 3em,
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
        *Kevin Cesenj*
      ],   
      [
        *Christina Eirini Christodoulou*
      ],
      [
        *Giusi Cozzi*
      ],
      [
        *Vecdi Gokmen*
      ],   
      [
        *Tebe Nigrelli*
      ],
      [
        *Lorena Pavel*
      ],
      [
        *Antonio Perazza*
      ],
      [
        *Emilio Vitali*
      ],
      [
        *Lia*
      ],
      [
        *Angelo Spahiu*
      ],
            
    )

    Following the kind direction of Postdoc Researcher Victor Buendìa

    #line(length: 100%)
  ]
]

#set heading(numbering: "1.a.")

= Introduction and motiation
what is reservior computing, why its interesting, applications, etc.


= Reservior dynamics

== Mathematical formulation
state update, leak rate, recurrent weights, input weights, readout 

== Dynamics overview
how the reservoir dynamics evolve, how they depend on the parameters, memory capacity, echo state property, 
(gaussain network case)
plots: neuron activaitios over time for J< 1, J=1, J>1



== Training Process
how to train readout, thermalisation then teacher forcing, ridge regression

= Reservoir dynamics on human connectome networks
goal is to combine our trianing pipeline with the human connectome data 

== Human Connectome Data and Network Analysis
graph theory for reservior networks, describe the data, how we process it, how we create the reservoir from it, etc.
plots: 3D brain network, degree distribution, etc.

== Training and prediction on Lorentz System
generating lorentz trajectories, training, evaluation of performance
plots: Lorenz trajectory, teacher-forcing fit, lambda-vs-error, 



= Analysis of reservoir dynamics on human connectome networks


