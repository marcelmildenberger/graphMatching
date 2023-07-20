import os
import pickle

import numpy as np
import pecanpy

from typing import List, Tuple, Union
from gensim.models import Word2Vec
from datetime import datetime

from gensim.models.callbacks import CallbackAny2Vec


class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = [0]

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss-self.losses[self.epoch-1]}')
        self.epoch += 1

class N2VEmbedder():

    def __init__(self, walk_length: int, n_walks: int, p: int, q: int, dim_embeddings: int, context_size: int, epochs,
                 seed: int = 1337, workers: int = -1):
        self.walk_length = walk_length
        self.n_walks = n_walks
        self.p = p
        self.q = q
        self.dim_embeddings = dim_embeddings
        self.context_size = context_size
        self.epochs = epochs
        self.seed = seed
        self.workers = workers if workers > 0 else os.cpu_count()

        self.model = None

        self.dev = False
        self.load_stored = False

    def train(self, data_dir: str):
        graph = pecanpy.pecanpy.SparseOTF(p=self.p, q=self.q, workers=self.workers, verbose=True, random_state=self.seed,
                                          extend=True)

        datname = "alice" if "alice" in data_dir else "eve"
        # Read the data  into the graph.
        graph.read_edg(data_dir, weighted=True, directed=False)
        # Generate the random walks
        if self.load_stored:
            with open("./dev/walks_"+datname+".pck", "rb") as f:
                walks = pickle.load(f)
        else:
            walks = graph.simulate_walks(num_walks=self.n_walks, walk_length=self.walk_length)

        if self.dev:
            with open("./dev/walks_"+datname+".pck", "wb") as f:
                pickle.dump(walks, f, protocol=5)


        # Run Word2Vec on the random walk to generate node2vec embeddings
        loss_logger = LossLogger()
        self.model = Word2Vec(
            walks, vector_size=self.dim_embeddings, window=self.context_size / 2, min_count=0, sg=1,
            workers=self.workers, epochs=self.epochs, compute_loss=True, callbacks=[loss_logger], seed=self.seed)
        #alpha=0.05, min_alpha=0.01, seed=self.seed)


    def save_model(self, path=None, filename=None):
        assert self.model is not None, "Model must be trained first"
        if path is None:
            path = "n2v_len%i_n%i_p%i_q%i_dim%i_con%i_e%i"
            path = path % (self.walk_length, self.n_walks, self.p, self.q, self.dim_embeddings, self.context_size,
                           self.epochs)
            path = os.path.join(".", path)

        if not os.path.isdir(path):
            os.makedirs(path)

        if filename is None:
            filename = "ex%i_%s.mod"
            now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            filename = filename % (len(self.model.wv), now)

        self.model.save(os.path.join(path, filename))

    def get_vectors(self, ordering: List[str] = None) -> np.ndarray:
        if ordering is None:
            ordering = [k for k in self.model.wv.key_to_index]
        embeddings = [self.model.wv.get_vector(k) for k in ordering]
        embeddings = np.stack(embeddings, axis=0)
        return embeddings, ordering

    def get_vector(self, key: Union[int, str]) -> np.ndarray:
        return self.model.wv.get_vector(key)
