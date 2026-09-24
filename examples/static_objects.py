"""Static objects are analyzed exactly as supplied, without hidden expansion."""
from topokit.core import simplicial, hyperdigraph, interaction
from topokit.builders import hyperdigraph as directed_builder, interaction as interaction_builder

sphere = simplicial.SimplicialComplex([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
print("Simplicial sphere H0..H2:", simplicial.homology(sphere).betti_numbers)
print("Simplicial sphere L2 nullity:", simplicial.laplacian(sphere, dimension=2).nullity)

directed = directed_builder.from_digraph([(0, 1), (1, 0)], vertices=(0, 1))
print("Reciprocal digraph H0..H2:", hyperdigraph.homology(directed).betti_numbers)

interacting = interaction_builder.from_complexes(sphere, sphere, max_dimension=2)
print("Two-factor sphere interaction H0..H2:", interaction.homology(interacting).betti_numbers)
print("Interaction L2 nullity:", interaction.laplacian(interacting, dimension=2).nullity)
