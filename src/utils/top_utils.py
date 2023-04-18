'''
Methods for calculating lower-dimensional persistent homology.
'''

import torch as th
import itertools
import numpy as np

class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices, device):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = th.arange(n_vertices, dtype=int, device=device)
        self.device = device

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistentHomologyCalculation:
    def __init__(self, device) -> None:
        self.device = device

    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices, self.device)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = th.argsort(edge_weights)

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        # Return empty cycles component
        return np.array(persistence_pairs), np.array([])


class TopologicalRegularizer(th.nn.Module):
    """Topologically regularizer."""

    def __init__(self, lam=0.1, device='cpu'):
        """Topologically Regularized Autoencoder.
        Args:
            lam: Regularization strength
            ae_kwargs: Kewords to pass to `ConvolutionalAutoencoder` class
            toposig_kwargs: Keywords to pass to `TopologicalSignature` class
        """
        super().__init__()
        self.lam = lam
        self.device = device
        self.topo_sig = TopologicalSignatureDistance(self.device)

    @staticmethod
    def _compute_distance_matrix(x, p=2):
        x_flat = x.view(x.size(0), -1)
        distances = th.linalg.vector_norm(x_flat[:, None] - x_flat, dim=2, ord=p)
        return distances

    def forward(self, models):
        """Compute the loss of the Topologically regularized autoencoder.
        Args:
            x: Input data
        Returns:
            Tuple of final_loss, (...loss components...)
        """
        model_distances = [model.rnn.weight for model in models]
        loss = sum([self.topo_sig(model1, model2) for model1, model2 in itertools.combinations(model_distances, 2)])
        return self.lam * loss

class TopologicalSignatureDistance(th.nn.Module):
    """Topological signature."""

    def __init__(self, device):
        """Topological signature computation.
        """
        super().__init__()
        self.device = device
        self.signature_calculator = PersistentHomologyCalculation(device)

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances)

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]
        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1)

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.
        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2
        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)
        sig1 = self._select_distances_from_pairs(distances1, pairs1)
        sig2 = self._select_distances_from_pairs(distances2, pairs2)
        distance = self.sig_error(sig1, sig2)
        
        return distance
