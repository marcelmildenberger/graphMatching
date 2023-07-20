import codecs, sys, time, math, argparse, ot
import numpy as np
from tqdm import trange
from .utils import *

def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


class WassersteinAligner:

    def __init__(self, maxload, reg_init, reg_ws, batchsize, lr, n_iter_init, n_iter_ws, n_epoch, vocab_size, lr_decay,
                 apply_sqrt, early_stopping, seed=42, verbose=True, min_epsilon=0.001):
        #np.random.seed(seed)
        self.maxload = maxload
        self.reg_init = reg_init
        self.reg_ws = reg_ws
        self.batchsize = batchsize
        self.lr = lr
        self.n_iter_ws = n_iter_ws
        self.n_iter_init = n_iter_init
        self.n_epoch = n_epoch
        self.vocab_size = vocab_size
        self.lr_decay = lr_decay
        self.apply_sqrt = apply_sqrt
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.min_epsilon = min_epsilon

        self.X = None
        self.Y = None


    def objective(self, R, n=1000):
        Xn, Yn = self.X[:n], self.Y[:n]
        C = -np.dot(np.dot(Xn, R), Yn.T)
        P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.9, stopThr=1e-3)
        return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n

    def solve_procrustes(self, R):
        assert self.X is not None and self.Y is not None, "Matrices must not be empty!"
        no_improvement = 0
        first_obj = -1
        prev_obj = float("inf")
        best_obj = float("inf")
        best_R = R
        for epoch in range(1, self.n_epoch + 1):
            if self.early_stopping > 0 and no_improvement >= self.early_stopping:
                if self.verbose:
                    print("Objective didn't improve for %i epochs. Stopping..." % self.early_stopping)
                    print("Improvement: %f" % (first_obj-best_obj))
                return best_R
            for _it in trange(1, self.n_iter_ws + 1, desc="Iteration", disable= not self.verbose):
                # sample mini-batch
                xt = self.X[np.random.permutation(self.X.shape[0])[:self.batchsize], :]
                yt = self.Y[np.random.permutation(self.Y.shape[0])[:self.batchsize], :]
                # compute OT on minibatch
                C = -np.dot(np.dot(xt, R), yt.T)
                P = ot.sinkhorn(np.ones(self.batchsize), np.ones(self.batchsize), C, self.reg_ws, stopThr=1e-3)
                # compute gradient
                G = - np.dot(xt.T, np.dot(P, yt))
                R -= self.lr / self.batchsize * G
                # project on orthogonal matrices
                U, s, VT = np.linalg.svd(R)
                R = np.dot(U, VT)
            self.lr *= self.lr_decay

            #obj = self.objective(R, n=min(1000, self.maxload))
            obj = self.objective(R, n=min(self.Y.shape[0], self.X.shape[0]))

            if first_obj == -1:
                first_obj = obj
            if obj < best_obj:
                best_obj = obj
                best_R = R

            if self.verbose or self.early_stopping > 0:
                if self.verbose:
                    print("epoch: %d  obj: %.3f" % (epoch, obj))

                if (prev_obj - obj) < self.min_epsilon:
                    no_improvement += 1
                else:
                    no_improvement = 0
                prev_obj = obj

        return best_R

    def convex_init(self, X = None, Y = None):
        if X is not None or Y is not None:
            self.X = X
            self.Y = Y

        # If the two matrices contain a different number of records, reduce the size to the smaller of the two
        # by random subsampling.
        if self.X.shape[0] < self.Y.shape[0]:
            X_c = self.X
            Y_c = self.Y[np.random.permutation(self.Y.shape[0])[:self.X.shape[0]], :]
        elif self.X.shape[0] > self.Y.shape[0]:
            X_c = self.X[np.random.permutation(self.X.shape[0])[:self.Y.shape[0]], :]
            Y_c = self.Y
        else:
            X_c = self.X
            Y_c = self.Y

        n, d = X_c.shape

        if self.apply_sqrt:
            X_c, Y_c = sqrt_eig(X_c), sqrt_eig(Y_c)
        K_X, K_Y = np.dot(X_c, X_c.T), np.dot(Y_c, Y_c.T)
        K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
        K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
        P = np.ones([n, n]) / float(n)
        for it in trange(1, self.n_iter_init + 1, disable=not self.verbose):
            G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
            q = ot.sinkhorn(np.ones(n), np.ones(n), G, self.reg_init, stopThr=1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = alpha * q + (1.0 - alpha) * P
        obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
        if self.verbose:
            print("Objective after convex initialization: " % obj)
        return procrustes(np.dot(P, X_c), Y_c).T, obj

    def align(self, src, tgt):
        self.X = src
        self.Y = tgt

        if self.verbose:
            print("\nComputing initial mapping with convex relaxation...")
        t0 = time.time()
        R0, _ = self.convex_init()
        if self.verbose:
            print("Done [%03d sec]" % math.floor(time.time() - t0))
            print("\nComputing mapping with Wasserstein Procrustes...")

        t0 = time.time()
        R = self.solve_procrustes(R0)
        if self.verbose:
            print("Done [%03d sec]" % math.floor(time.time() - t0))

        return R
