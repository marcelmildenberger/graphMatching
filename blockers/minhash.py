from random import seed, randint
import binascii


class MinHashLSH():
    """A class that implements a min-hashing locality sensitive hashing (LSH)
       approach to be used for blocking the plain-text q-grams sets in order to
       prevent a full-pair-wise comparison of all q-gram set pairs.
    """

    def __init__(self, lsh_band_size, lsh_num_band, random_seed=None):
        """Initialise the parameters for min-hashing LSH including generating
           random values for hash functions.

           Input arguments:
             - lsh_band_size  The length of the min-hash bands.
             - lsh_num_band   The number of LSH bands.
             - random_seed    If not None then initalise the random number
                              generator with this seed value.

           Output:
             - This method does not return anything.

           LSH min-hashing follows the code provided here:
            https://github.com/chrisjmccormick/MinHash/blob/master/ \
                  runMinHashExample.py

           The probability for a pair of sets with Jaccard sim 0 < s <= 1 to be
           included as a candidate pair is (with b = lsh_num_band and
           r = lsh_band_size, i.e. the number of rows/hash functions per band) is
           (Leskovek et al., 2014, page 89):

             p_cand = 1- (1 - s^r)^b

           Approximation of the 'threshold' of the S-curve (Leskovek et al., 2014,
           page 90) is: t = (1/k)^(1/r).
        """

        if (random_seed != None):
            seed(random_seed)

        # Calculate error probabilities for given parameter values
        #
        assert lsh_num_band > 1, lsh_num_band
        assert lsh_band_size > 1, lsh_band_size

        self.lsh_num_band = lsh_num_band
        self.lsh_band_size = lsh_band_size
        self.num_hash_funct = lsh_band_size * lsh_num_band  # Total number needed

        b = float(lsh_num_band)
        r = float(lsh_band_size)
        t = (1.0 / b) ** (1.0 / r)

        s_p_cand_list = []
        for i in range(1, 10):
            s = 0.1 * i
            p_cand = 1.0 - (1.0 - s ** r) ** b
            assert 0.0 <= p_cand <= 1.0
            s_p_cand_list.append((s, p_cand))

        print('Initialise LSH blocking using Min-Hash')
        print('  Number of hash functions: %d' % (self.num_hash_funct))
        print('  Number of bands:          %d' % (lsh_num_band))
        print('  Size of bands:            %d' % (lsh_band_size))
        print('  Threshold of s-curve:     %.3f' % (t))
        print('  Probabilities for candidate pairs:')
        print('   Jacc_sim | prob(cand)')
        for (s, p_cand) in s_p_cand_list:
            print('     %.2f   |   %.5f' % (s, p_cand))
        print()

        max_hash_val = 2 ** 31 - 1  # Maximum possible value a CRC hash could have

        # We need the next largest prime number above 'maxShingleID'.
        # From here:
        # http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
        #
        self.next_prime = 4294967311

        # Random hash function will take the form of: h(x) = (a*x + b) % c
        # where 'x' is the input value, 'a' and 'b' are random coefficients, and
        # 'c' is a prime number just greater than max_hash_val
        #
        # Generate 'num_hash_funct' coefficients
        #
        coeff_a_set = set()
        coeff_b_set = set()

        while (len(coeff_a_set) < self.num_hash_funct):
            coeff_a_set.add(randint(0, max_hash_val))
        while (len(coeff_b_set) < self.num_hash_funct):
            coeff_b_set.add(randint(0, max_hash_val))
        self.coeff_a_list = sorted(coeff_a_set)
        self.coeff_b_list = sorted(coeff_b_set)
        assert self.coeff_a_list != self.coeff_b_list

    # ---------------------------------------------------------------------------

    def hash_q_gram_set(self, q_gram_set):
        """Min-hash the given set of q-grams and return a list of hash signatures
           depending upon the Min-hash parameters set during the class
           initialisation.

           Input arguments:
             - q_gram_set  The q-gram set to be hashed.

           Output:
             -  band_hash_sig_list  A list with the min-hash signatures for the
                                    input q-gram set.
        """

        next_prime = self.next_prime
        coeff_a_list = self.coeff_a_list
        coeff_b_list = self.coeff_b_list
        lsh_band_size = self.lsh_band_size
        lsh_num_band = self.lsh_num_band

        crc_hash_set = set()

        for q_gram in q_gram_set:  # Hash the q-grams into 32-bit integers
            crc_hash_set.add(binascii.crc32(q_gram.encode()) & 0xffffffff)

        assert len(q_gram_set) == len(crc_hash_set)  # Check no collision

        # Now generate all the min-hash values for this q-gram set
        #
        min_hash_sig_list = []

        for h in range(self.num_hash_funct):

            # For each CRC hash value (q-gram) in the q-gram set calculate its Min-
            # hash value for all 'num_hash_funct' functions
            #
            min_hash_val = next_prime + 1  # Initialise to value outside range

            for crc_hash_val in crc_hash_set:
                hash_val = (coeff_a_list[h] * crc_hash_val + coeff_b_list[h]) % \
                           next_prime
                min_hash_val = min(min_hash_val, hash_val)

            min_hash_sig_list.append(min_hash_val)

        # Now split hash values into bands and generate the list of
        # 'lsh_num_band' hash values used for blocking
        #
        band_hash_sig_list = []

        start_ind = 0
        end_ind = lsh_band_size
        for band_num in range(lsh_num_band):
            band_hash_sig = min_hash_sig_list[start_ind:end_ind]
            assert len(band_hash_sig) == lsh_band_size
            start_ind = end_ind
            end_ind += lsh_band_size
            band_hash_sig_list.append(band_hash_sig)

        return band_hash_sig_list
