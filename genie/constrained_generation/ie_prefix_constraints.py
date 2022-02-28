from collections import deque
from typing import Dict, List

import numpy as np
import torch

from .trie import Trie


def get_information_extraction_prefix_allowed_tokens_fn_hf(
    model,
    sentences: List[str],
    subject_token="sub",
    relation_token="rel",
    object_token="obj",
    end_of_triple_token="et",
    start_of_tag="<",
    end_of_tag=">",
    relations_trie: Trie = None,
    entities_trie: Trie = None,
    bos_as_first_token_generated=False,
):
    return _get_information_extraction_prefix_allowed_tokens_fn_fairseq(
        lambda x: model.tokenizer.encode(x),
        lambda x: model.tokenizer.decode(torch.tensor(x), skip_special_tokens=True),
        model.tokenizer.bos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.eos_token_id,
        len(model.tokenizer),
        sentences,
        subject_token,
        relation_token,
        object_token,
        end_of_triple_token,
        start_of_tag,
        end_of_tag,
        relations_trie,
        entities_trie,
        bos_as_first_token_generated,
    )


def get_information_extraction_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences: List[str],
    subject_token="sub",
    relation_token="rel",
    object_token="obj",
    end_of_triple_token="et",
    start_of_tag="<",
    end_of_tag=">",
    relations_trie: Trie = None,
    entities_trie: Trie = None,
    bos_as_first_token_generated=False,
):
    return _get_information_extraction_prefix_allowed_tokens_fn_fairseq(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        sentences,
        subject_token,
        relation_token,
        object_token,
        end_of_triple_token,
        start_of_tag,
        end_of_tag,
        relations_trie,
        entities_trie,
        bos_as_first_token_generated,
    )


def _get_information_extraction_prefix_allowed_tokens_fn_fairseq(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    sentences: List[str],
    subject_token="sub",
    relation_token="rel",
    object_token="obj",
    end_of_triple_token="et",
    start_of_tag="<",
    end_of_tag=">",
    relations_trie: Trie = None,
    entities_trie: Trie = None,
    bos_as_first_token_generated=False,
):
    full_codes = {
        n: encode_fn(
            " {}{}{}".format(start_of_tag, c, end_of_tag)
        )  # encoder adds bos_token_id and eos_token_id in the beginning and the end of encoding respectivelly
        for n, c in zip(
            (
                "subject_token",
                "relation_token",
                "object_token",
                "end_of_entity_token",
            ),
            (
                subject_token,
                relation_token,
                object_token,
                end_of_triple_token,
            ),
        )
    }

    # check that lengths match, and that the start and end of tag tokens are the same
    l = []
    s = []
    e = []
    for n, c in full_codes.items():
        l.append(len(c))
        s.append(c[1])
        e.append(c[-2])

    assert np.all(np.array(l) == l[0])
    assert np.all(np.array(s) == s[0])
    assert np.all(np.array(e) == e[0])

    codes = {n: full_codes[n][2] for n in full_codes}
    tag_codes = set(codes[k] for k in codes)

    codes["start_of_tag"] = s[0]
    codes["end_of_tag"] = e[0]

    codes["EOS"] = eos_token_id
    codes["BOS"] = bos_token_id

    status_codes = ["ob", "s", "r", "o"]
    status_next_token_name = ["subject_token", "relation_token", "object_token", "end_of_entity_token"]

    if sentences is not None:
        sent_origs = [[codes["EOS"]] + encode_fn(sent)[1:] for sent in sentences]
    else:
        sent_origs = []

    def get_status(sent):
        """Returns the generation setting – mention generation, entity generation or outside"""
        status = 0

        i = 0
        while i < len(sent) - 2:
            if sent[i] == codes["start_of_tag"] and sent[i + 1] in tag_codes and sent[i + 2] == codes["end_of_tag"]:
                status += 1

            i += 1

        status = status % 4

        return status, status_codes[status]

    def get_last_tag_pointer(sent):
        """Assumes that the last tag is fully generated i.e. <tag_name>"""
        i = len(sent) - 2

        while i >= 0:
            if sent[i] == codes["start_of_tag"] and sent[i + 1] in tag_codes and sent[i + 2] == codes["end_of_tag"]:
                return i, i + 2

            i -= 1

        return None

    def prefix_allowed_tokens_fn(batch_id, sent):
        """Sent is the thus far generated sequence of ids acting as output.
        Batch_id is the idx of the sentence that we are generating the output for."""
        sent = sent.tolist()

        # TODO: Figure out when and why the generation doesn't end after EOS is generated.
        # TODO: If the next two lines are removed, output contains many "EOS EOS EOS..." at the end.
        if len(sent) > 1 and sent[-1] == codes["EOS"]:
            return []

        # Force the generation of BOS as a first token to be generated
        # Necessary if the model is trained with [eos bos ... eos] as target
        if bos_as_first_token_generated and len(sent) == 1:
            return [codes["BOS"]]

        status, status_code = get_status(sent)
        if len(sent_origs) == 0:
            sent_orig = None
        else:
            sent_orig = sent_origs[batch_id]

        # ---- IF inside a tag ----
        # return the next status tag if the start tag was generated last
        if len(sent) > 0 and sent[-1] == codes["start_of_tag"]:
            return [codes[status_next_token_name[status]]]

        # return closing tag if the start tag and the status tag have been generated
        if len(sent) > 1 and sent[-2] == codes["start_of_tag"]:
            if sent[-1] in tag_codes:
                return [codes["end_of_tag"]]
            else:
                return []
        # -------------------------

        # ---- If outside of a tag ----
        # Get allowed tokens
        allowed_tokens = get_allowed_tokens(sent, sent_orig, status_code)
        return allowed_tokens

    def get_allowed_tokens(sent, sent_orig, status_code):
        if status_code == "ob":
            allowed_tokens = [codes["start_of_tag"], codes["EOS"]]
        elif status_code == "s":
            allowed_tokens = _get_allowed_tokens(sent, sent_orig, entities_trie)
        elif status_code == "r":
            allowed_tokens = _get_allowed_tokens(sent, sent_orig, relations_trie)
        elif status_code == "o":
            allowed_tokens = _get_allowed_tokens(sent, sent_orig, entities_trie)
        else:
            raise RuntimeError

        return allowed_tokens

    def _get_allowed_tokens(sent, sent_orig, trie):
        pointer_start, pointer_end = get_last_tag_pointer(sent)

        allowed_tokens = trie.get(sent[pointer_end + 1 :])

        if codes["EOS"] in allowed_tokens:
            allowed_tokens.remove(codes["EOS"])
            allowed_tokens.append(codes["start_of_tag"])

        return allowed_tokens

    return prefix_allowed_tokens_fn
