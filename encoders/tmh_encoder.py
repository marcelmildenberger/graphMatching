# tabminhash.py - Implementation of Duncan Smith' tabulation min-hash based
# PPRL approach to encode strings into bit arrays.
#
# Peter Christen, June to September 2018
# Youzhe Heng, October 2020
# Jochen Schäfer, July 2023
# -----------------------------------------------------------------------------

import hashlib
import itertools
import random
import bitarray
import numpy as np
from tqdm import trange, tqdm
from sklearn.metrics import pairwise_distances_chunked


# =============================================================================

def reduce_func(D_chunk, start):
    global tuids
    res = []
    for row in D_chunk:
        tmp = []
        for j, val in enumerate(row[0:start]):
            tmp.append((tuids[start], tuids[j], val))
        res.append(tmp)
        start += 1
    return res


def reduce_func_sim(D_chunk, start):
    global tuids
    res = []
    for row in D_chunk:
        tmp = []
        for j, val in enumerate(row[0:start]):
            tmp.append((tuids[start], tuids[j], 1 - val))
        res.append(tmp)
        start += 1
    return res

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

        print('Generating list with %d tabulation hash tables, each with %d ' % \
              (num_hash_bits, num_tables) + 'tables, each table with %d ' % \
              (2 ** key_len) + 'entries (key length %d bits and value length %d' % \
              (key_len, val_len) + ' bits)')
        print()

        tab_hash_table_list = []  # The 'num_hash_bits' tabulation hash tables

        if random_seed != None:
            random.seed(random_seed)

        for hash_bit in trange(num_hash_bits, desc="Initializing hash tables", disable=not self.verbose):

            # The list of hash tables to be generated for a certain bit position,
            # where each will be a dictionary with bit strings of length 'key_len'
            # as keys and values being bit arrays of length 'val_len'
            #
            bit_pos_tab_hash_table = []

            # Generate all binary values from 0..0 to 1...1 of length 'key_len' bits
            # and for each generate a random bit array of length 'val_len'.

            # Use itertools to generate all sub-sets of positions to be set to 1
            #
            all_bit_pos = range(key_len)
            key_bit_array_list = []

            for sub_set_len in range(key_len + 1):
                for pos_sub_set in itertools.combinations(all_bit_pos, sub_set_len):

                    # Generate an empty (all zero) bit array of length 'key_len' and set
                    # all bits with the indices given in the generated position sub set
                    #
                    table_key_bit_array = bitarray.bitarray(key_len)
                    table_key_bit_array.setall(0)

                    for pos in pos_sub_set:
                        table_key_bit_array[pos] = 1

                    # Note bitarrays cannot be dictionary keys, so convert into strings
                    #
                    key_bit_array_list.append(table_key_bit_array.to01())

            assert len(key_bit_array_list) == 2 ** key_len, \
                (len(key_bit_array_list), 2 ** key_len)

            for t in range(num_tables):

                table_dict = {}

                # For each unique bit position set generate a bit array of length
                # 'val_len' with randomly set bits to 0 or 1 (50% likelihood each)
                #
                for table_key_bit_array_str in key_bit_array_list:
                    table_val_bit_array = bitarray.bitarray(val_len)
                    table_val_bit_array.setall(0)

                    for pos in range(val_len):
                        table_val_bit_array[pos] = random.choice([0, 1])

                    table_dict[table_key_bit_array_str] = table_val_bit_array

                assert len(table_dict) == len(key_bit_array_list)

                bit_pos_tab_hash_table.append(table_dict)

            tab_hash_table_list.append(bit_pos_tab_hash_table)

        assert len(tab_hash_table_list) == num_hash_bits

        self.tab_hash_table_list = tab_hash_table_list

    # ---------------------------------------------------------------------------

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
        tab_hash_table = self.tab_hash_table_list[bit_pos]

        # Generate the bit array for the input string based on the given hash
        # function
        #
        encoded_in_str = in_str.encode()
        hash_hex_digest = self.hash_funct(encoded_in_str).hexdigest()
        hash_int = int(hash_hex_digest, 16)
        hash_bit_array_str = bin(hash_int)[2:]  # Remove leading '0b'
        # print hash_bit_array_str

        # Now take the lowest 'key_len' bits from the hash bit array string and
        # use them to get the initial tabulation hash value
        #
        tab_hash_table_key = hash_bit_array_str[-key_len:]

        # Get the random bit pattern from the first table
        #
        tab_hash_bit_array = tab_hash_table[0][tab_hash_table_key].copy()

        # And XOR with the remaining extracted tabulation hashing table values
        #
        for t in range(1, num_tables):
            tab_hash_table_key = hash_bit_array_str[-key_len * (t + 1):-key_len * t]

            # XOR (^) of table hash values
            #
            tab_hash_bit_array ^= tab_hash_table[t][tab_hash_table_key]

        return tab_hash_bit_array

    # ---------------------------------------------------------------------------

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

        # Generate the final bit array to be returned
        #
        q_gram_bit_array = bitarray.bitarray(num_hash_bits)

        for bit_pos in range(num_hash_bits):

            min_hash_val = None  # Only keep the minimum min hash value

            for q_gram in q_gram_set:

                tab_hash_bit_array = self.__get_tab_hash(q_gram, bit_pos)

                if (min_hash_val == None):
                    min_hash_val = tab_hash_bit_array
                else:
                    min_hash_val = min(min_hash_val, tab_hash_bit_array)

            # Get the last bit of the smallest tabulation hash value and insert into
            # the final bit array
            #
            q_gram_bit_array[bit_pos] = min_hash_val[-1]

        return q_gram_bit_array

    def encode_and_compare(self, data, uids, metric, sim=True):
        available_metrics = ["braycurtis", "canberra", "chebyshev", "cityblock", "correlation", "cosine", "dice",
                             "euclidean", "hamming", "jaccard", "jensenshannon", "kulczynski1", "mahalanobis",
                             "matching",
                             "minkowski", "rogerstanimoto", "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
                             "sqeuclidean", "yule"]
        assert metric in available_metrics, "Invalid similarity metric. Must be one of " + str(available_metrics)

        data = ["".join(d).replace(" ","") for d in data]
        # Split each string in the data into a list of qgrams to process
        data = [[b[i:i + self.ngram_size] for i in range(len(b) - self.ngram_size + 1)] for b in data]

        offset = 0
        enc = []
        for i in tqdm(data, desc="Encoding", disable=not self.verbose, total=len(uids)):
            enc.append(self.__encode_q_gram_set(i))

        enc = np.stack([list(barr) for barr in enc]).astype(bool)
        global tuids
        tuids = uids
        if sim:
            pw_metrics = pairwise_distances_chunked(enc, metric=metric, n_jobs=-1, reduce_func=reduce_func_sim)
        else:
            pw_metrics = pairwise_distances_chunked(enc, metric=metric, n_jobs=-1, reduce_func=reduce_func)
        pw_metrics_long = []
        for i in pw_metrics:
            pw_metrics_long += i
        pw_metrics_long = [item for row in pw_metrics_long for item in row]

        return pw_metrics_long