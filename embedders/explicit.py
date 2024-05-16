from collections import defaultdict
from operator import itemgetter
import statistics
import math
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize


# Return Method for the defaultdict
def ret_zero():
    return 0


class ExplicitEmbedder():

    def __init__(self, graph, encodings, uids, min_component_size=None, verbose=False):
        self.encodings = encodings
        self.uids = uids
        self.min_component_size = min_component_size
        self.G = None
        self.num_edges = None
        self.num_nodes = None
        self.max_degree = None
        self.degree_centr = None
        self.betweenness_centr = None
        self.length_dict = {}
        self.node_frequencies = {}
        self.verbose = verbose

        if type(graph) == str:
            self.__load(graph)
        else:
            self.G = graph

        if min_component_size is not None:
            self.__enforce_component_size()

        self.__init_dicts()

    def __load(self, edgelist_dir):
        """
        Loads the similarity graph from a weighted edgelist to NetworkX.
        :return:
        """
        self.G = nx.read_weighted_edgelist(edgelist_dir)
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()
        _, self.max_degree = sorted(self.G.degree, key=itemgetter(1))[-1]
        self.max_log_degree = math.floor(math.log(self.max_degree, 2))

        if self.verbose:
            print("Loaded Graph with %i nodes, %i edges and maximum node degree of %i." % (
                self.num_nodes, self.num_edges, self.max_degree))

    def __enforce_component_size(self):

        conn_comp = list(nx.connected_components(self.G))
        small_comp = [comp for comp in conn_comp if len(comp) < self.min_component_size]
        drop = [node for comp in small_comp for node in comp]
        if self.verbose:
            print("Similarity Graph has %i connected components, %i of which are smaller than the minimum connected "
                  "component size %i. Dropping %i nodes." % (len(conn_comp), len(small_comp),
                                                             self.min_component_size, len(drop)))

        for d in drop:
            self.G.remove_node(d)
            self.num_nodes -= 1

        if self.verbose:
            print("Done.")

    def __init_dicts(self):
        """
        Initializes the dictionaries that contain information on node length (i.e. the number of 1-bits in the
        Bloom Filter or the Integer/q-gram Set size) and node frequency (i.e. how often a given q-gram/integer set or
        BF encoding occurs in the encoded database)
        :return:
        """

        # If the encodings are stored in a Numpy array, set the length to the number of non-zero elements.
        # For Bit Arrays this corresponds to the hamming weight, for other encodings it is equal to the length.
        if type(self.encodings) in [np.ndarray, np.array]:
            for uid, enc in zip(self.uids, self.encodings):
                self.length_dict[uid] = len(np.where(enc != 0)[0])
        # If the encodings are a list of lists, set the node length to the number of elements in the respective list.
        elif type(self.encodings) == list:
            for uid, enc in zip(self.uids, self.encodings):
                self.length_dict[uid] = len(set(enc))
        else:
            raise "Invalid encoding datatype"

        # We have to turn the encodings into hashable datatypes, i.e. tuples, to use them as keys in dictionaries.
        freq_dict = defaultdict(ret_zero)

        # If the encodings are bit arrays store the position of the 1-bits
        if (type(self.encodings) in [np.ndarray, np.array]) and (self.encodings.dtype == np.bool_):
            for enc in self.encodings:
                freq_dict[tuple(np.where(enc == 1)[0])] = freq_dict[tuple(np.where(enc == 1)[0])] + 1
            # Store the frequencies by UID
            for uid, enc in zip(self.uids, self.encodings):
                self.node_frequencies[uid] = freq_dict[tuple(np.where(enc == 1)[0])]
        else:
            for enc in self.encodings:
                freq_dict[tuple(set(enc))] = freq_dict[tuple(set(enc))] + 1
            # Store the frequencies by UID
            for uid, enc in zip(self.uids, self.encodings):
                self.node_frequencies[uid] = freq_dict[tuple(set(enc))]

        if self.verbose:
            print("Computing betweenness centrality. This may take a while...")

        self.betweenness_centr = nx.betweenness_centrality(self.G, weight="weight")

        if self.verbose:
            print("Computing degree centrality. This may take a while...")

        self.degree_centr = nx.degree_centrality(self.G)

    def get_vectors(self, ordering=None, hist_features=None):

        if hist_features is None:
            hist_features = self.max_log_degree

        if ordering is None:
            ordering = list(self.G.nodes())

        # Initialize the feature array
        num_feat = 11 + (2 * (hist_features + 1))
        feat_array = np.zeros((len(ordering), num_feat), dtype=np.float32)

        if self.verbose:
            print("Done. Computing %i features for %i nodes" % (num_feat, len(ordering)))

        for i, node in tqdm(enumerate(ordering), total=len(ordering), disable=not self.verbose):
            j = 0
            # Node Length
            feat_array[i, j] = self.length_dict[node]
            j += 1
            # Node Frequency
            feat_array[i, j] = self.node_frequencies[node]
            j += 1
            connected_edges = self.G.edges(node)
            # Node degree
            feat_array[i, j] = len(connected_edges)
            j += 1
            edgeweights = [self.G.get_edge_data(edge[0], edge[1])["weight"] for edge in connected_edges]
            if len(edgeweights) > 0:
                # Maximum edge weight
                feat_array[i, j] = max(edgeweights)
                j += 1
                # Minimum edgeweight
                feat_array[i, j] = min(edgeweights)
                j += 1
                # Average Edgeweight
                feat_array[i, j] = statistics.mean(edgeweights)
                j += 1
            else:
                feat_array[i, j] = 0
                j += 1
                feat_array[i, j] = 0
                j += 1
                feat_array[i, j] = 0
                j += 1

            # Standard deviation of edgeweights
            if len(edgeweights) > 1:
                feat_array[i, j] = statistics.stdev(edgeweights)
            else:
                feat_array[i, j] = 0
            j += 1
            # Generate Egonet
            node_egograph = nx.ego_graph(self.G, node)
            neighbor_set = set(node_egograph.nodes)
            # Generate 2-Neighborhood
            two_neighborhood = nx.single_source_shortest_path_length(self.G, node, cutoff=2)
            # Restruct to nodes that are exactly two hops away
            two_neighborhood = [key for key, val in two_neighborhood.items() if val == 2]
            # Egonet Density
            feat_array[i, j] = nx.density(node_egograph)
            j += 1
            # Egonet Degree (Num of edges that connect the nodes inside the egonet to nodes outside egonet)
            egonet_degree = 0
            for neighbor in node_egograph:
                # Get 1-Neighborhood of Neighbor (Number of neighbors equals degree of node as we have at most one edge
                # between each node pair and the graph is undirected.
                neighbor_one_neighborhood = nx.single_source_shortest_path_length(self.G, neighbor, cutoff=1)
                neighbor_one_neighborhood = set([key for key, val in neighbor_one_neighborhood.items() if val == 1])
                neighbor_log_degree = math.floor(math.log(len(neighbor_one_neighborhood), 2))
                # Increase the respective field in the "histogram"
                feat_array[i, 11 + neighbor_log_degree] += 1
                # For each node that is a neighbor of the neighbor, but not in the original node's egonet there must
                # exist a node connecting the neigbor to the node outside the egonet
                egonet_degree += len(neighbor_one_neighborhood) - len(
                    neighbor_one_neighborhood.intersection(neighbor_set))
            for two_neighbor in two_neighborhood:
                two_neighbor_log_degree = math.floor(math.log(self.G.degree(two_neighbor), 2))
                feat_array[i, 11 + hist_features + 1 + two_neighbor_log_degree] += 1
            feat_array[i, j] = egonet_degree
            j += 1
            feat_array[i, j] = self.betweenness_centr[node]
            j += 1
            feat_array[i, j] = self.degree_centr[node]

        # Normalize feature vectors
        # Note: Column-Wise normalization
        feat_array = normalize(feat_array, axis=0, norm='max')
        return feat_array, ordering
