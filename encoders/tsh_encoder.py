# colminhash.py - Implementation of a column-based vector hashing approach
# for PPRL to encode strings into integer hash vectors.
#
# Based on discussions at ICDM 2018 (Singapore)
# Rainer Schnell, Peter Christen, and Thilina Ranbaduge
#
# October 2020
# -----------------------------------------------------------------------------

# Notes from email Peter to Rainer and Thilina, 20 November 2018:
#
# Ideas for improved encoding based on individual hash-function BFs and
# column-wise tabulation hashing
#
# Each hash function generates one BF, where each such BFs encodes all
# q-grams from a record (i.e. for a single BF each q-gram is hashed only once
# using one of the k hash functions).
#
# This will give one bit matrix per record, with k rows and l columns. How to
# select optimal / suitable k and l we will need to investigate (ideally
# proof of optimality would  be good).
#
# So we will have 2^k possible column bit patterns in this matrix, where 2^k
# >> l to ensure (ideally) each column can have an unique bit pattern.
#
# We use a column bit pattern to seed a PRNG, and take the first (or xth)
# random number (in range (min,max) to represent the column, resulting in a
# vector of l numbers.
#
# We now have:
# - Two q-gram sets that are the same result in the same vector of random
#   numbers.
# - Two q-gram sets with 1 q-gram different can have a maximum of k (from l)
#   different random number.
# - Two q-gram sets with n q-gram different can have a maximum of n*k (from l)
#   different random number.
# (to check/proof if this is correct)
#
# This allows us to calculate similarities.
#
# We will need to check how this relates to locality sensitive hashing and
# Duncan's approach. Also, we need to analyse the privacy of this - are there
# frequent patterns that could be exploited? I don't think so. Each random
# number in the generated vectors encodes on column in a bit matrix and thus
# several q-grams generated by different hash functions.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Peter, 20181204, possible attack ideas:
# - For each hash vector position, count how often the value is -1 (i.e. input
#   bit array is 0...0, and count the frequency of each hash integer at each
#   position. These frequencies reveal information about the frequency of the
#   input bit patterns -but is thisuseful?
# - How else to attack?
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Peter, 20181204
# - for blocking, we can do min-hash by only considering positive values
# - is there work on sparse min-hash?
# Thilina, 05092019
# - update the code to generate unique interger values for each column
# - hence there will not be any collision between the interger values

# Adapted in 2023 by Jochen Schäfer
import os
import random
import string
import numpy as np
import bitarray
import hashlib
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances_chunked

from tqdm import trange, tqdm

# =============================================================================

HASH_FUNCT = hashlib.sha1
def q_gram_dice_sim(q_gram_set1, q_gram_set2):
    """calculate the dice similarity between two given sets of q-grams.
       Dice sim(A,B)= 2 x number of common elements of A and B
                      ----------------------------------------
                      number of elements in A + number of elements in B
       returns a similarity value between 0 and 1.
    """

    num_common_q_gram = len(q_gram_set1.intersection(q_gram_set2))

    q_gram_dice_sim = (2.0 * num_common_q_gram) / \
                      (len(q_gram_set1) + len(q_gram_set2))

    return q_gram_dice_sim


# -----------------------------------------------------------------------------

def q_gram_jacc_sim(q_gram_set1, q_gram_set2):
    """calculate the jaccard similarity between two given sets of q-grams.
       Jaccard sim(A,B) = |A intersection B|
                          ------------------
                              |A Union B|
       returns a value between 0 and 1.
    """

    q_gram_intersection_set = q_gram_set1.intersection(q_gram_set2)
    q_gram_union_set = q_gram_set1.union(q_gram_set2)

    q_gram_jacc_sim = float(len(q_gram_intersection_set) / len(q_gram_union_set))

    return q_gram_jacc_sim


def compute_metrics(inds, cache, uids, metric, sim):
    tmp = np.zeros((len(inds),3), dtype=float)
    pos = 0
    for i, j in inds:
        j_enc = cache[uids[j]]
        i_enc = cache[uids[i]]
        if metric == "jaccard":
            val = q_gram_jacc_sim(i_enc, j_enc)
        else:
            val = q_gram_dice_sim(i_enc, j_enc)

        if not sim:
            val = 1 - val
        tmp[pos] = np.array([uids[i], uids[j], val])
        pos += 1
    return tmp


def make_inds(i_vals, numex):
    tmp1 = []
    for i in i_vals:
        tmp2 = []
        for j in range(i + 1, numex):
            tmp2.append(np.array([i, j], dtype=int))
        if len(tmp2)>0:
            tmp1.append(np.vstack(tmp2))
    return np.vstack(tmp1)

class TSHEncoder():
    """A class that implements a column-based vector hashing approach for PPRL
       to encode strings into integer hash value sets.
    """

    def __init__(self, num_hash_funct, num_hash_col, ngram_size, rand_mode='PNG', secret=None, verbose=True, workers=-1):
        """To initialise the class we need to set the number of hash functions,
           the number of hash columns (bit arrays) to generate, the maximum
           integer for the random numbers to be generated, and a counter for
           which random value to be used in the final hash vector for each column.

           Input arguments:
             - num_hash_funct  The number of hash functions to be used to hash
                               each q-gram of an input q-grams set.
             - num_hash_col    The number of hash columns, i.e. bit arrays, to be
                               generated.
             - rand_mode       The random integer number generation mode, this can
                               either be generated using SHA function combines with
                               a modulo operation (MOD) or using a pseudo number
                               generator (PNG).
           Output:
             - This method does not return anything.
        """

        assert num_hash_funct > 1, num_hash_funct
        assert num_hash_col > 1, num_hash_col
        assert rand_mode in ['PNG', 'MOD'], rand_mode

        self.num_hash_funct = num_hash_funct
        self.num_hash_col = num_hash_col
        self.rand_mode = rand_mode
        self.ngram_size = ngram_size
        self.salt = ''.join(random.choice(string.ascii_letters) for i in range(32)) if secret is None else secret
        self.verbose = verbose
        self.workers = os.cpu_count() if workers == -1 else workers

    # ---------------------------------------------------------------------------

    def encode(self, q_gram_set):
        """Apply column-based vector hashing on the given input q-gram set and
           generate a hash value set which is returned.

           Input arguments:
             - q_gram_set  The set of q-grams (strings) to be encoded.

           Output:
             - hash_set  A set of hash values representing the q-gram set.
        """

        num_hash_funct = self.num_hash_funct
        num_hash_col = self.num_hash_col
        rand_mode = self.rand_mode

        max_rand_int = 2 ** num_hash_funct

        # Number of hash columns minus 1 to be used to get the range of columns
        # that can be selected to hash a q-gram
        #
        num_hash_col_m1 = num_hash_col - 1

        # Initialise one bit array per hash column, each of length 'num_hash_funct'
        #
        col_ba_list = []

        for i in range(num_hash_col):
            col_ba = bitarray.bitarray(num_hash_funct)
            col_ba.setall(0)
            col_ba_list.append(col_ba)

        # Step 1: For each hash function, use one row in the bit arrays and encode
        # all q-grams of the given input set
        #
        for q_gram in q_gram_set:

            # Use q-gram itself to seed random number generator
            #
            random_seed = random.seed(q_gram)

            # Because we used the q-gram to seed the random number generator, the
            # sequence of integers generated in the following loop will be the same
            # for the same q-gram
            #
            for row_index in range(num_hash_funct):
                col_index = random.randint(0, num_hash_col_m1)

                col_ba_list[col_index][row_index] = 1

        # It does not matter if several column bit arrays are the same because we
        # add the column number to ensure the seed values for the random number
        # generator are unique even if the bit patterns are the same

        ## Check if all column bit arrays are different
        ##
        # col_ba_dict = {}
        # for col_ba in col_ba_list:
        #  col_ba_str = col_ba.to01()
        #  col_ba_dict[col_ba_str] = col_ba_dict.get(col_ba_str, 0) + 1
        # if (len(col_ba_dict) < len(col_ba_list)):
        #  print '*** Warning: %d column bit arrays are the same! ***' % \
        #        (len(col_ba_list) - len(col_ba_dict)+1)
        #  for (col_ba_str, col_ba_count) in col_ba_dict.iteritems():
        #    if (col_ba_count > 1):
        #      print '***         %s occurs %d times' % (col_ba_str, col_ba_count)

        # Step 2: Use each column bit array that has at least one 1-bit as the
        # seed of the random number generator to generate an integer hash value
        # for this column, concatenate with the column number, and add to the set
        # of hash values for this q-gram set
        #
        hash_set = set()

        for (col_index, col_ba) in enumerate(col_ba_list):

            # If there are no 1-bits then we do not use this column
            #
            if (col_ba.any() == True):  # At least one 1-bit

                col_index_str = str(col_index)

                # Important, add the column number to the seed to make sure the same
                # bit pattern at different positions does not result in the same
                # random seed value
                #
                rand_str = col_ba.to01() + col_index_str + str(self.salt)
                random.seed(rand_str)
                # random.seed(col_ba.to01())

                hash_val = 0

                if rand_mode == 'PNG':
                    rand_min = col_index * max_rand_int
                    rand_max = rand_min + max_rand_int
                    hash_val = random.randint(rand_min, rand_max)

                elif rand_mode == 'MOD':
                    encoded_rand_str = rand_str.encode()
                    hex_str = HASH_FUNCT(encoded_rand_str).hexdigest()
                    int_val = int(hex_str, 16)

                    hash_val = (col_index * max_rand_int) + int(int_val % max_rand_int)

                # Concatenate the column index with the random integer value to
                # create the final hash value for this column
                #
                # hash_set.add(col_index_str+'-'+str(hash_val))
                # assert hash_val not in hash_set, (hash_set, hash_val)
                hash_set.add(hash_val)

        return hash_set

    def encode_and_compare(self, data, uids, metric, sim=True):
        available_metrics = ["jaccard", "dice"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)
        uids = [float(u) for u in uids]
        data = ["".join(d).replace(" ", "").lower() for d in data]
        # Split each string in the data into a list of qgrams to process
        data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]

        parallel = Parallel(n_jobs=self.workers)

        output_generator = parallel(delayed(self.encode) (i) for i in data)
        cache = {}
        print("Caching")
        for i, enc in enumerate(output_generator):
            cache[uids[i]] = enc
        del output_generator

        if self.workers > 1:
            numex = len(uids)
            print("Enumerating")
            # dim = ((len(uids)*len(uids))-len(uids))//2
            # inds = np.zeros((dim,2), dtype=int)
            # pos = 0
            # for i in range(numex):
            #     for j in range(i+1, numex):
            #         inds[pos] = np.array([i,j])
            #         pos += 1

            output_generator = parallel(delayed(make_inds)(i, numex) for i in np.array_split(np.arange(numex),self.workers*2))
            inds = np.vstack(output_generator)

            #chunksize = len(inds) // self.workers
            inds = np.array_split(inds, self.workers)
            # if chunksize < 1:
            #     self.workers = 1
            # else:
            #     print("Splitting")
            #     inds = split(inds, chunksize)

        if self.workers > 1:
            print("Calculating")
            pw_metrics = parallel(delayed(compute_metrics)(i, cache, uids, metric, sim) for i in inds)
            return np.vstack(pw_metrics)

        else:
            dim = ((len(uids)*len(uids))-len(uids))//2
            pw_metrics = np.zeros((dim, 3), dtype=float)
            ind = 0
            for i, uid in tqdm(enumerate(uids), desc="Encoding", total=len(data), disable=not self.verbose):
                i_enc = cache[uid]
                for j, q_j in enumerate(data[i + 1:]):
                    j_enc = cache[uids[j + i + 1]]
                    if metric == "jaccard":
                        val = q_gram_jacc_sim(i_enc, j_enc)
                    else:
                        val = q_gram_dice_sim(i_enc, j_enc)

                    if not sim:
                        val = 1 - val

                    pw_metrics[ind] = np.array([uid, uids[j + i + 1], val])
                    ind += 1
            return pw_metrics
