import config as C
import random
import csv

def get_triplet(spk_to_utts):
    """Get a triplet of anchor/pos/neg samples."""
    pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    # Retry if too few positive utterances.
    while len(spk_to_utts[pos_spk]) < 2:
        pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    anchor_utt, pos_utt = random.sample(spk_to_utts[pos_spk], 2)
    neg_utt = random.sample(spk_to_utts[neg_spk], 1)[0]
    return (anchor_utt, pos_utt, neg_utt)


def get_csv_spk_to_utts(csv_file):
    """Get the dict from speaker to list of utterances from CSV file."""
    spk_to_utts = dict()
    with open(csv_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue
            spk = row[0].strip()
            utt = row[1].strip()
            if spk not in spk_to_utts:
                spk_to_utts[spk] = [utt]
            else:
                spk_to_utts[spk].append(utt)
    return spk_to_utts


def main():
    print('done')






if __name__ == '__main__':

    main()