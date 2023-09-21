import numpy as np
import random
class Simulator:
    """
    Defines core functionality for simulating similarities of encodings according to different probability
    distributions.
    """

    def __init__(self, n, range_sim=[0, 1]):
        """
        Constructor for simulators
        :param n: The number of records to simulate
        :param range_sim: Range of possible similarity values
        :param range_true: Range of similarity values for matching records
        """
        self.n = n
        self.range_sim = range_sim
        self.selected_alice = None
        self.selected_eve = None
        self.selected_overlap = None

    def calc_sizes(self, drop_from, overlap):
        if drop_from == "Alice":
            self.selected_alice = random.sample(range(self.n), int(overlap * self.n))
            self.selected_eve = range(self.n)
        elif drop_from == "Eve":
            self.selected_alice = range(self.n)
            self.selected_eve = random.sample(range(self.n), int(overlap * self.n))
        elif drop_from == "Both":
            # See main.py Line 111 for explanation
            overlap_count = int(-(overlap * self.n / (overlap - 2)))
            available = list(range(self.n))
            self.selected_overlap = random.sample(available, overlap_count)
            available = [i for i in available if i not in self.selected_overlap]
            self.selected_alice = random.sample(available, int((self.n - overlap_count) / 2))
            available = [i for i in available if i not in self.selected_alice]
            self.selected_alice += self.selected_overlap
            self.selected_eve = self.selected_overlap + available
        else:
            raise Exception("drop_from must be Alice, Eve or Both")

    def to_pairwise(self, alice_data, eve_data):
        alice_dim = ((len(self.selected_alice)**2) - len(self.selected_alice)) // 2
        pw_alice = np.zeros((alice_dim, 3), dtype=float)

        eve_dim = ((len(self.selected_eve)**2) - len(self.selected_eve)) // 2
        pw_eve = np.zeros((eve_dim, 3), dtype=float)

        ind = 0
        for i, uid_i in enumerate(self.selected_alice):
            for uid_j in self.selected_alice[i+1:]:
                pw_alice[ind] = np.array([uid_i, uid_j, alice_data[uid_i,uid_j]])
                ind += 1

        ind = 0
        for i, uid_i in enumerate(self.selected_eve):
            for uid_j in self.selected_eve[i+1:]:
                pw_eve[ind] = np.array([uid_i, uid_j, eve_data[uid_i,uid_j]])
                ind += 1

        return pw_alice, pw_eve
