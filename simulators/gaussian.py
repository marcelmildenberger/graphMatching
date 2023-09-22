import numpy as np

from .simulator import Simulator


class GaussianSimulator (Simulator):

    def simulate_data(self, overlap=1, drop_from="Alice"):
        rng = np.random.default_rng()
        self.calc_sizes(drop_from, overlap)

        # Initialize similarity array with zeroes
        center = sum(self.range_sim)/2
        sd = abs(self.range_sim[0] - self.range_sim[1])/4
        alice_data = rng.normal(center, sd, size=(self.n, self.n))
        # Parameters for noise distribution have been empirically determined as
        # center (mu) 0.002528 and standard deviation 0.034849
        noise = rng.normal(0.002528, 0.034849, size=(self.n, self.n))
        eve_data = alice_data + noise
        alice_data[alice_data>1] = 1
        alice_data[alice_data<0] = 0
        eve_data[eve_data>1] = 1
        eve_data[eve_data<0] = 0

        return self.to_pairwise(alice_data, eve_data)


if __name__ == "__main__":
    testSim = GaussianSimulator(10, range_sim=[0,1])
    testAlice, testEve = testSim.simulate_data(overlap=0.5, drop_from="Eve")