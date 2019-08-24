# DependencyParser
Neural Net Dependency Parser


This is an implementation of the arc-standard dependency parser that uses a feed-forward neural network to predict dependency relations for a given state.

For example, given the input sentence below, the parser will derive a graph with dependency relations (e.g., det., advmod, etc.). The parser utilizes a neural network and a training corpus to predict dependency relations at each configuration.

![Dependency Parser](https://nlp.stanford.edu/software/nndep-example.png)
