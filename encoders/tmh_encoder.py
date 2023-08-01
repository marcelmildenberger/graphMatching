# tabminhash.py - Implementation of Duncan Smith' tabulation min-hash based
# PPRL approach to encode strings into bit arrays.
#
# Peter Christen, June to September 2018
# Youzhe Heng, October 2020
# Performance optimizations by Jochen Schäfer, July 2023
# -----------------------------------------------------------------------------
import random
import numpy as np
from tqdm import trange, tqdm
from sklearn.metrics import pairwise_distances_chunked

def jaccard_sim(D_chunk, start):
    global tuids
    global n_bits
    res = []
    for row in D_chunk:
        tmp = []
        for j, val in enumerate(row[0:start]):
            # Compute the approximated jaccard similarity
            val = max(0.0, 1.0 - 2.0 * float(val) / n_bits)
            tmp.append((tuids[start], tuids[j], val))
        res.append(tmp)
        start += 1
    return res


def jaccard_dist(D_chunk, start):
    global tuids
    global n_bits
    res = []
    for row in D_chunk:
        tmp = []
        for j, val in enumerate(row[0:start]):
            # Compute the approximated jaccard similarity
            val = max(0.0, 1.0 - 2.0 * float(val) / n_bits)
            tmp.append((tuids[start], tuids[j], 1 - val))
        res.append(tmp)
        start += 1
    return res

# =============================================================================
class TMHEncoder():
    """A class that implements tabulation based min-hash encoding of string
       values into bit arrays for privacy-preserving record linkage, as proposed
       in:

       Secure pseudonymisation for privacy-preserving probabilistic record
       linkage, Duncan Smith, Journal of Information Security and Applications,
       34 (2017), pages 271-279.
    """

    def __init__(self, num_hash_bits, num_tables, key_len, val_len, hash_funct,
                 ngram_size, random_seed=42, verbose=False):
        """To initialise the class for each of the 'num_hash_bits' a list of
           'num_tables' tabulation hash tables need to be generated where keys are
           all bit patterns (strings) of length 'len_key' and values are random
           bit strings of length 'val_len' bits.

           Input arguments:
             - num_hash_bits  The total number of bits to be generated when a
                              value is being hashed (the length of the final bit
                              array to be generated).
             - num_tables     The number of tables to be generated.
             - key_len        The length of the keys into these tables as a number
                              of bits (where each table will contain 2^key_len
                              elements).
             - val_len        The length of the random bit strings to be generated
                              for each table entry as a number of bits.
             - hash_funct     The actual hash function to be used to generate a
                              bit string for an input string value (q-gram).
             - random_seed    To ensure repeatability the seed to initialise the
                              pseudo random number generator. If set to None then
                              no seed it set.

           Output:
             - This method does not return anything.
        """

        assert num_hash_bits > 1, num_hash_bits
        assert num_tables > 1, num_tables
        assert key_len > 1, key_len

        self.num_hash_bits = num_hash_bits
        self.num_tables = num_tables
        self.key_len = key_len
        self.hash_funct = hash_funct
        self.ngram_size = ngram_size
        self.verbose = verbose

        if verbose:
            print('Generating list with %d tabulation hash tables, each with %d ' % \
                  (num_hash_bits, num_tables) + 'tables, each table with %d ' % \
                  (2 ** key_len) + 'entries (key length %d bits and value length %d' % \
                  (key_len, val_len) + ' bits)')
            print()

        if random_seed != None:
            random.seed(random_seed)

        # Create multi-dimendsinal array of random bits
        self.hashtables = np.random.randint(2, size=(num_hash_bits, num_tables, 2 ** key_len, val_len))

    def __get_tab_hash(self, in_str, bit_pos):
        """Generate a tabulation hash for the given input string based on the
       tabulation hash tables for the given bit position, by retrieving
       'num_tables' tabulation hash values from table entries based on the
       hashed input string value.

       Input arguments:
         - in_str   The string to be hashed.
         - bit_pos  The bit position for which the tabulation hash should be
                    generated.

       Output:
         - tab_hash_bit_array  A bit array generated from the corresponding
                               tabulation hash tables.
        """
        assert bit_pos >= 0 and bit_pos < self.num_hash_bits
        num_tables = self.num_tables
        key_len = self.key_len

        # Get the tabulation hash table for the desired bit position
        #
        tab_hashtable = self.hashtables[bit_pos]
        # Generate the bit array for the input string based on the given hash
        # function
        #

        hash_hex_digest = self.hash_funct(in_str.encode()).hexdigest()
        hash_int = int(hash_hex_digest, 16)
        hash_bit_array_str = bin(hash_int)[2:]  # Remove leading '0b'
        # Now take the lowest 'key_len' bits from the hash bit array string and
        # use them to get the initial tabulation hash value
        #
        tab_hashtable_key = hash_bit_array_str[-key_len:]
        # Get the random bit pattern from the first table
        #
        tab_hash_bit_array = tab_hashtable[0][int(tab_hashtable_key, 2)].copy()
        # And XOR with the remaining extracted tabulation hashing table values
        #
        for t in range(1, num_tables):
            tab_hashtable_key = hash_bit_array_str[-key_len * (t + 1):-key_len * t]

            # XOR (^) of table hash values
            #
            tab_hash_bit_array ^= tab_hashtable[t][int(tab_hashtable_key, 2)]

        return tab_hash_bit_array

    def __encode_q_gram_set(self, q_gram_set):
        """Apply tabulation based min hashing on the given input q-gram set and
           generate a bit array which is returned.

           Input arguments:
             - q_gram_set  The set of q-grams (strings) to be encoded.

           Output:
             - q_gram_bit_array  The bit array encoding the given q-gram set.
        """

        num_hash_bits = self.num_hash_bits  # Short-cuts
        get_tab_hash = self.__get_tab_hash
        q_gram_bit_array = np.zeros(self.num_hash_bits, dtype=int)

        for bit_pos in range(num_hash_bits):

            min_hash_val = None  # Only keep the minimum min hash value
            min_hash_arr = None

            for q_gram in q_gram_set:

                tab_hash_bit_array = self.__get_tab_hash(q_gram, bit_pos)
                # Calculate the integer value of the hash in the bit array
                cur_hash_val = tab_hash_bit_array.dot(1 << np.arange(tab_hash_bit_array.shape[-1] - 1, -1, -1))
                if (min_hash_val == None):
                    min_hash_val = cur_hash_val
                    min_hash_arr = tab_hash_bit_array
                else:
                    if cur_hash_val < min_hash_val:
                        min_hash_val = cur_hash_val
                        min_hash_arr = tab_hash_bit_array

                # Get the last bit of the smallest tabulation hash value and insert into
                # the final bit array
                #
            q_gram_bit_array[bit_pos] = min_hash_arr[-1]

        return q_gram_bit_array

    def encode_and_compare(self, data, uids, metric, sim=True):
        available_metrics = ["jaccard"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        data = ["".join(d).replace(" ","") for d in data]
        # Split each string in the data into a list of qgrams to process
        data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]

        offset = 0
        enc = []
        for i in tqdm(data, desc="Encoding", disable=not self.verbose, total=len(uids)):
            enc.append(self.__encode_q_gram_set(i))

        enc = np.stack(enc).astype(bool)
        global tuids
        tuids = uids
        global n_bits
        n_bits = self.num_hash_bits
        if sim:
            # There is no sklearn/scipy implementation for the approximated jaccard metric used in TabMinHash.
            # However, we still want to use sklearn's pairwise_distances_chunked because of performance and
            # to avoid running out of memory. We can compute the cityblock/manhatten distance though, which, in case
            # of binary inputs, is equal to the number of different bits. The actual computation of approximate jaccard
            # is perfomed in the reduce function.
            # This is a somewhat hacky solution, but I currently can't think of a better one.
            # Jochen, 2023
            pw_metrics = pairwise_distances_chunked(enc, metric="cityblock", n_jobs=-1, reduce_func=jaccard_sim)
        else:
            pw_metrics = pairwise_distances_chunked(enc, metric="cityblock", n_jobs=-1, reduce_func=jaccard_dist)
        pw_metrics_long = []
        for i in pw_metrics:
            pw_metrics_long += i
        pw_metrics_long = [item for row in pw_metrics_long for item in row]

        return pw_metrics_long