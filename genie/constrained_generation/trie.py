from collections import defaultdict
import jsonlines
import pickle
import os


def get_trie_from_strings(
    string_iterable,
    add_leading_space_flag=True,
    remove_leading_bos=True,
    output_folder_path=None,
    trie_name=None,
    tokenizer=None,
):
    assert (output_folder_path is None and trie_name is None) or (
        output_folder_path is not None and trie_name is not None
    )
    from tqdm import tqdm

    if tokenizer is None:
        from transformers import BartTokenizer

        tokenizer = BartTokenizer.from_pretrained("martinjosifoski/genie-rw")

    if add_leading_space_flag:
        leading_space = lambda x: f" {x}"
    else:
        leading_space = lambda x: x

    if remove_leading_bos:
        leading_bos = lambda x: x[1:]
    else:
        leading_bos = lambda x: x

    encode_func = lambda x: leading_bos(tokenizer(leading_space(x))["input_ids"])
    trie = Trie([encode_func(uniq_name) for uniq_name in tqdm(sorted(string_iterable))])

    if output_folder_path is not None:
        trie.dump(output_folder_path=output_folder_path, file_name=trie_name, string_iterable=string_iterable)

    return trie


class Trie(object):
    def __init__(self, sequences):
        """sequences is a list of lists,
        each of which corresponds to a sequence of tokens encoded by the tokenizer"""
        next_sets = defaultdict(list)  # a dict that returns an empty list when the key is not in it
        for seq in sequences:
            if len(seq) > 0:
                next_sets[seq[0]].append(seq[1:])

        self._leaves = {k: Trie(v) for k, v in next_sets.items()}
        # for the leaves of the trie _leaves == {}

    def get(self, indices):  # indices holds the list of vocabulary tokens that constitute the current prefix
        if len(indices) == 0:  # if we haven't generated anything so far: return all possible starting tokens
            return list(self._leaves.keys())
        elif indices[0] not in self._leaves:
            # if the currently leading token (and by extension the prefix) isn't eligible: return an empty list
            return []
        else:
            return self._leaves[indices[0]].get(indices[1:])  # take the trie that corresponds to the

    def dump(self, output_folder_path, file_name, string_iterable=None):
        pickle.dump(self, open(os.path.join(output_folder_path, f"{file_name}.pickle"), "wb"), protocol=4)

        if string_iterable is not None:
            with jsonlines.open(os.path.join(output_folder_path, f"{file_name}_original_strings.jsonl"), "w") as writer:
                writer.write_all(string_iterable)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            trie = pickle.load(f)

        return trie
