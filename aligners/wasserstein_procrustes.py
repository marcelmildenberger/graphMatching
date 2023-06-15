import codecs, sys, time, math, argparse, ot
import numpy as np
from tqdm import trange
from .utils import *
import networkx as nx

def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))


class WassersteinAligner():

    def __init__(self, maxload, reg_init, reg_ws, batchsize, lr, n_iter_init, n_iter_ws, n_epoch, vocab_size, lr_decay,
                 apply_sqrt, early_stopping, seed=42, verbose=True):
        np.random.seed(seed)
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
        prev_obj = float("inf")
        if self.early_stopping > 0 and no_improvement >= self.early_stopping:
            print("Objective didn't improve for %i epochs. Stopping..." % self.early_stopping)
            return R
        for epoch in trange(1, self.n_epoch + 1, desc="Epoch"):
            for _it in trange(1, self.n_iter_ws + 1, desc="Iteration", leave=False):
                # sample mini-batch
                xt = self.X[np.random.permutation(self.maxload)[:self.batchsize], :]
                yt = self.Y[np.random.permutation(self.maxload)[:self.batchsize], :]
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

            if self.verbose or self.early_stopping > 0:
                obj = self.objective(R, n=min(1000, self.maxload))
                if self.verbose:
                    print("epoch: %d  obj: %.3f" % (epoch, obj))
                if obj >= prev_obj:
                    no_improvement += 1
                elif obj < prev_obj:
                    no_improvement = 0
                prev_obj = obj

        return R


    def convex_init(self):
        n, d = self.X.shape
        if self.apply_sqrt:
            self.X, self.Y = sqrt_eig(self.X), sqrt_eig(self.Y)
        K_X, K_Y = np.dot(self.X, self.X.T), np.dot(self.Y, self.Y.T)
        K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
        K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
        P = np.ones([n, n]) / float(n)
        for it in trange(1, self.n_iter_init + 1):
            G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
            q = ot.sinkhorn(np.ones(n), np.ones(n), G, self.reg_init, stopThr=1e-3)
            alpha = 2.0 / float(2.0 + it)
            P = alpha * q + (1.0 - alpha) * P
        obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
        print(obj)
        return procrustes(np.dot(P, self.X), self.Y).T

    def align(self, src, tgt):
        self.X = src
        self.Y = tgt

        src = src[:self.maxload]
        tgt = tgt[:self.maxload]

        print("\nComputing initial mapping with convex relaxation...")
        t0 = time.time()
        R0 = self.convex_init()
        print("Done [%03d sec]" % math.floor(time.time() - t0))

        print("\nComputing mapping with Wasserstein Procrustes...")
        t0 = time.time()
        R = self.solve_procrustes(R0)
        print("Done [%03d sec]" % math.floor(time.time() - t0))

        self.X = self.X / np.linalg.norm(self.X, 2, 1).reshape([-1, 1])
        self.Y = self.Y / np.linalg.norm(self.Y, 2, 1).reshape([-1, 1])
        self.Y = np.dot(self.Y, R.T)

        return self.X, self.Y
