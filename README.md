# DependencyParser
Neural Net Dependency Parser


This is an implementation of the arc-standard dependency parser that uses a feed-forward neural network to predict dependency relations for a given state.

For example, given the input sentence 'This time around, they're moving even faster.', the parser will derive a graph with dependency relations (e.g., det., advmod, etc.). The parser utilizes a neural network and a training corpus to predict dependency relations at each configuration.

<p align="center">
  <img width="460" height="300" src="https://nlp.stanford.edu/software/nndep-example.png">
</p>
