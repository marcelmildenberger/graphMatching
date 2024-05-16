import csv
from typing import Sequence
from collections import defaultdict
import statistics

def read_tsv(path: str, header: bool = True, as_dict: bool = False, delim: str = "\t") -> Sequence[Sequence[str]]:
    data = {} if as_dict else []
    uid = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delim)
        if header:
            next(reader)
        for row in reader:
            if as_dict:
                assert len(row) == 3, "Dict mode only supports rows with two values + uid"
                data[row[0]] = row[1]
            else:
                data.append(row[:-1])
                uid.append(row[-1])
    return data, uid


def save_tsv(data, path: str, delim: str = "\t", mode="w"):
    with open(path, mode, newline="") as f:
        csvwriter = csv.writer(f, delimiter=delim)
        csvwriter.writerows(data)


def create_blocks(uids, ngrams, minhash, verbose=True):
    def not_present():
        return None

    def empty_list():
        return []

    block_count = 0
    signature_to_block_id = defaultdict(not_present)
    uids_to_blocks = defaultdict(empty_list)
    blocks_to_uids = defaultdict(empty_list)

    for uid, ngr in zip(uids, ngrams):
        signatures = minhash.hash_q_gram_set(set(ngr))
        for s in signatures:
            block_id = signature_to_block_id[tuple(s)]
            if block_id is None:
                block_id = block_count
                signature_to_block_id[tuple(s)] = block_id
                block_count += 1
            # Adds the block to the list of blocks thes uid belongs to
            uid_blocks = uids_to_blocks[uid]
            uid_blocks.append(block_id)
            uids_to_blocks[uid] = uid_blocks
            # Adds the uid to the list of uids a block contains

            block_uids = blocks_to_uids[block_id]
            block_uids.append(uid)
            blocks_to_uids[block_id] = block_uids

    if verbose:
        blocksizes = []
        for v in blocks_to_uids.values():
            blocksizes.append(len(v))
        print("Created %i distinct blocks. Size info: Min %i, Max %i, Mean %0.3f, Median %i" % (
        len(blocksizes), min(blocksizes), max(blocksizes), statistics.mean(blocksizes), statistics.median(blocksizes)))
    return uids_to_blocks, blocks_to_uids


def simulate_blocking(enc_data, uids_to_block):
    keep = []
    for i in range(len(enc_data)):
        common_blocks = set(uids_to_block[str(int(enc_data[i][0]))]).intersection(set(uids_to_block[str(int(enc_data[i][1]))]))
        if len(common_blocks)>0:
            keep.append(i)
    return keep