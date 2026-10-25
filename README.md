

This repo is home to a novel extension of Adaptive ES-HyperNEAT called Adaptive Specialized-ES-HyperNEAT. The following abstract overviews Adaptive SES-HyperNEAT's novelty:


  Neuroevolution mimics Darwinian evolution to evolve AI neural networks from scratch for control problems such as robotic control. Many features of biological brains have yet to be implemented in neuroevolution algorithms and may improve their ability to evolve intelligence.
  
  One such feature is the evolution of functionally distinct neuron types fundamental to biological brains’ intricate circuitry. Prior neuroevolution algorithms encode the circuitry and plasticity (adaptive characteristics) of neural networks as a geometric pattern, but this does not provide the discrete functional organization underlying biological motifs, neuroplasticity, neuromodulation, topography, etc. Therefore, this study introduces a novel extension of Adaptive ES-HyperNEAT called Adaptive Specialized-ES-HyperNEAT which evolves multiple ‘branches’ of connection weight output nodes in the encoding and evolves sets of Hebbian coefficients for respective branches’ connections to share, mirroring the organized functional diversity of biological brains.
  
  Comparing the performance of Adaptive SES-HyperNEAT to Adaptive ES-HyperNEAT on a continuous T-maze task, Adaptive SES-HyperNEAT demonstrated more exploration and generalization but less exploitation. This may have been due to the algorithm’s less precise encoding of plasticity (multiple connections shared the same Hebbian coefficients). However, the novel algorithm’s significantly increased speciation sheds light on the fragility of neuroevolution’s results under the unideal manual tuning of parameters. Provided that this is improved for further development of NEAT’s extensions, Adaptive SES-HyperNEAT opens the door to implementing significant features of biological neural networks such as functionally-specific global neuromodulation, advancing our understanding of how neuroevolution algorithms can be improved with future inspiration from biology for evolving more intelligent neural networks.



---------
Below is the description of Pureples:

PUREPLES - Pure Python Library for ES-HyperNEAT
===============================================

About
-----
This is a library of evolutionary algorithms with a focus on neuroevolution, implemented in pure python, depending on the [neat-python](https://github.com/CodeReclaimers/neat-python) implementation. It contains a faithful implementation of both HyperNEAT and ES-HyperNEAT which are briefly described below.

**NEAT** (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks.  
**HyperNEAT** (Hypercube-based NEAT) is a method developed by Kenneth O. Stanley utilizing NEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain.  
**ES-HyperNEAT** (Evolvable-substrate HyperNEAT) is a method developed by Sebastian Risi and Kenneth O. Stanley utilizing HyperNEAT. It is a technique for evolving large-scale neural networks using the geometric regularities of the task domain. In contrast to HyperNEAT, the substrate used during evolution is able to evolve. This rids the user of some initial work and often creates a more suitable substrate.

The library is extensible in regards to easy transition between experimental domains.

Getting started
---------------
This section briefly describes how to install and run experiments.  

### Installation Guide
First, make sure you have the dependencies installed: `numpy`, `neat-python`, `graphviz`, `matplotlib` and `gym`.  
All the above can be installed using [pip](https://pip.pypa.io/en/stable/installing/).  
Next, download the source code and run `setup.py` (`pip install .`) from the root folder. Now you're able to use **PUREPLES**!

### Experimenting
How to experiment using NEAT will not be described, since this is the responsibility of the `neat-python` library.

Setting up an experiment for **HyperNEAT**:
* Define a substrate with input nodes and output nodes as a list of tuples. The hidden nodes is a list of lists of tuples where the inner lists represent layers. The first list is the topmost layer, the last the bottommost.
* Create a configuration file defining various NEAT specific parameters which are used for the CPPN.
* Define a fitness function setting the fitness of each genome. This is where the CPPN and the ANN is constructed for each generation - use the `create_phenotype_network` method from the `hyperneat` module.
* Create a population with the configuration file made in (2).
* Run the population with the fitness function made in (3) and the configuration file made in (2). The output is the genome solving the task or the one closest to solving it.

Setting up an experiment for **ES-HyperNEAT**:
Use the same setup as HyperNEAT except for:
* Not declaring hidden nodes when defining the substrate.
* Declaring ES-HyperNEAT specific parameters.
* Using the `create_phenotype_network` method residing in the `es_hyperneat` module when creating the ANN.

If one is trying to solve an experiment defined by the [OpenAI Gym](https://gym.openai.com/) it is even easier to experiment. In the `shared` module a file called `gym_runner` is able to do most of the work. Given the number of generations, the environment to run, a configuration file, and a substrate, the relevant runner will take care of everything regarding population, fitness function etc.

Please refer to the sample experiments included for further details on experimenting. 

