import jsonlines

import pandas as pd
import numpy as np

from collections import Counter

from .triplet_utils import TripletUtils


def get_relations_and_relation_occurrence_counts_in_dataset(dataset):
    r = []

    for ds in dataset:
        ids = get_rels_in_sample(ds)
        r.extend(ids)

    c = Counter(r)
    return r, c


def get_rels_in_sample(data_point):
    relations = list([r for _, r, _ in TripletUtils.convert_text_sequence_to_text_triples(data_point["trg"])])
    return relations


def get_descriptions_for_relations_in_mapping(relation_mapping):
    import copy
    import pywikibot
    from tqdm import tqdm

    id2info = copy.deepcopy(relation_mapping.id2surface_form)
    no_desc = 0

    for wikidata_id in tqdm(id2info.keys()):
        site = pywikibot.Site("wikidata", "wikidata")
        repo = site.data_repository()
        item = pywikibot.PropertyPage(repo, "{}".format(wikidata_id))
        data = item.get(force=True)

        if "en" in data["descriptions"]:
            label_desc = data["descriptions"]["en"]
        else:
            label_desc = None
            no_desc += 1

        id2info[wikidata_id]["en_description"] = label_desc

    print(f"All except `{no_desc}` relations have been assigned a description")


class AnnotationAnalysisObject(object):
    def __init__(self, processed_data, per_triple_allowed_statuses={"title"}, per_sample_allowed_statuses=None):
        rel_id2matched = {}
        rel_id2unmatched = {}
        rel_id2unmatched_lit = {}
        rel_id2rel_name = {}

        all_triples = []

        for sample in processed_data:
            valid_sample = True
            if per_sample_allowed_statuses is not None:
                if sample["output"][0]["instance_matching_status"] not in per_sample_allowed_statuses:
                    valid_sample = False

            text_triples = sample["output"][0]["non_formatted_surface_output"]
            if "non_formatted_wikidata_id_output" in sample["meta_obj"]:
                code_triples = sample["meta_obj"]["non_formatted_wikidata_id_output"]
            else:
                code_triples = sample["output"][0]["non_formatted_wikidata_id_output"]
            triple_matching_statuses = sample["output"][0]["non_formatted_triples_match_status"]

            for code_triple, text_triple, triple_matching_status in zip(
                code_triples, text_triples, triple_matching_statuses
            ):
                # All relations are expected to be matched
                assert (
                    text_triple[1] is not None
                ), f"Relation with wikidata ID `{code_triple}` has not been matched with a textual label"

                if code_triple[1] not in rel_id2rel_name:
                    rel_id2rel_name[code_triple[1]] = text_triple[1]

                code_text_pair = (code_triple, text_triple)
                if valid_sample and triple_matching_status in per_triple_allowed_statuses:
                    matched = rel_id2matched.get(code_triple[1], [])
                    matched.append(code_text_pair)
                    rel_id2matched[code_triple[1]] = matched
                else:
                    s, r, o = code_triple
                    if s.startswith("Q") and o.startswith("Q"):
                        unmatched = rel_id2unmatched.get(code_triple[1], [])
                        unmatched.append(code_text_pair)
                        rel_id2unmatched[code_triple[1]] = unmatched
                    else:
                        unmatched = rel_id2unmatched_lit.get(code_triple[1], [])
                        unmatched.append(code_text_pair)
                        rel_id2unmatched_lit[code_triple[1]] = unmatched

                all_triples.append(text_triple[1])

        # Sanity checks
        c_all_triples = Counter(all_triples)
        for rel_id in rel_id2rel_name:
            assert (
                len(rel_id2matched.get(rel_id, []))
                + len(rel_id2unmatched.get(rel_id, []))
                + len(rel_id2unmatched_lit.get(rel_id, []))
                == c_all_triples[rel_id2rel_name[rel_id]]
            )

        assert (
            len(set(rel_id2unmatched.keys()).intersection(rel_id2unmatched_lit.keys())) == 0
        ), f"The intersection between relations unmatched due to literal and due to missing data is not zero and equals to: {set(rel_id2unmatched.keys()).intersection(rel_id2unmatched_lit.keys())}"

        print(f"{len(rel_id2unmatched_lit.keys())} relations rely on literal entities")

        # exclude literal relatations in unmatched entity analysis
        rel_ids = np.array(list(set(list(rel_id2rel_name.keys())) - set(rel_id2unmatched_lit.keys())))
        sorted_idx = np.argsort([len(rel_id2unmatched.get(rel_id, [])) for rel_id in rel_ids])[::-1]
        rel_ids = rel_ids[sorted_idx]

        _total = [c_all_triples[rel_id2rel_name[rel_id]] for rel_id in rel_ids]
        _matched = [len(rel_id2matched.get(rel_id, [])) for rel_id in rel_ids]
        _unmatched = [len(rel_id2unmatched.get(rel_id, [])) for rel_id in rel_ids]
        _names = [rel_id2rel_name[rel_id] for rel_id in rel_ids]

        df = pd.DataFrame(
            {
                "relation_ids": rel_ids,
                "relation name": _names,
                "total": _total,
                "matched": _matched,
                "unmatched": _unmatched,
                "unmatched_portion": [um / (m + um) for m, um in zip(_matched, _unmatched)],
            }
        )
        print(
            f"There are `{sum(df['matched'] != 0)}` relations that have non-zero occurence (and `{sum(df['matched'] == 0)}` without any valid triples)"
        )

        print(
            f"From the total `{sum(_total)} triples`, {sum(_unmatched) * 100 / (sum(_matched) + sum(_unmatched))} % (i.e. {sum(_unmatched)}) remained unmatched"
        )

        self.df = df.sort_values("matched", ascending=False).reset_index(drop=True)
        self.rel_id2matched = rel_id2matched
        self.rel_id2unmatched = rel_id2unmatched
        self.rel_id2unmatched_lit = rel_id2unmatched_lit
        self.rel_id2rel_name = rel_id2rel_name

    def plot_matching_results_scatter(self, top_k=None, _range=None, fig_size=(22, 4)):
        assert (top_k is None or _range is None) and (top_k is not None or _range is not None)
        import matplotlib.pyplot as plt

        if top_k is not None:
            p_data = self.df[:top_k]
        else:
            p_data = self.df.iloc[_range[0] : _range[1]]

        rel_names = p_data["relation name"]
        fig, axs = plt.subplots(figsize=fig_size)

        axs.scatter(range(0, len(rel_names) * 3, 3), p_data["matched"], label="matched")
        axs.scatter(range(0, len(rel_names) * 3, 3), p_data["unmatched"], label="unmatched")

        plt.xticks(range(0, len(rel_names) * 3, 3), rel_names, rotation=90)
        plt.legend()
        plt.show()


class WikidataAnnotator(object):
    def __init__(self, ent_mapping, rel_mapping, query_wikidata, allow_labels):
        self.ent_mapping = ent_mapping
        self.rel_mapping = rel_mapping
        self.query_wikidata = query_wikidata
        self.allow_labels = allow_labels

    def annotate_kilt_dataset(self, data):
        for processed_obj in data:
            if "non_formatted_wikidata_id_output" in processed_obj["meta_obj"]:
                triples = processed_obj["meta_obj"]["non_formatted_wikidata_id_output"]
            else:
                triples = processed_obj["output"][0]["non_formatted_wikidata_id_output"]
                processed_obj["meta_obj"]["non_formatted_wikidata_id_output"] = triples

            non_formatted_triples_match_status = []
            non_formatted_surface_output = []
            non_formatted_surface_output_provenance = []
            instance_status = "title"

            for t in triples:
                match_status, wikidata_id_form, surface_form, provenance = TripletUtils.process_triple_of_ids(
                    t, self.ent_mapping, self.rel_mapping, self.query_wikidata, self.allow_labels
                )

                non_formatted_triples_match_status.append(match_status)
                non_formatted_surface_output.append(surface_form)
                non_formatted_surface_output_provenance.append(provenance)

                # All relations are expected to be matched
                assert (
                    surface_form[1] is not None
                ), f"Relation with wikidata ID `{surface_form[1]}` has not been matched with a textual label"

                if instance_status != "no_match" and match_status == "label":
                    instance_status = match_status
                elif match_status == "no_match":
                    instance_status = match_status
                else:
                    assert match_status == "title" or match_status == "label"

            if instance_status == "no_match":
                answer = None
            else:
                answer = TripletUtils.triples_to_output_format(non_formatted_surface_output)

            output_obj = {
                "non_formatted_triples_match_status": non_formatted_triples_match_status,
                "non_formatted_surface_output": non_formatted_surface_output,
                "non_formatted_surface_output_provenance": non_formatted_surface_output_provenance,
                "instance_matching_status": instance_status,
                "answer": answer,
            }

            processed_obj["output"] = [output_obj]

            if "instance_matching_status" in processed_obj:
                del processed_obj["instance_matching_status"]

        return data


class WikidataID2SurfaceForm(object):
    def __init__(self, path):
        self.path = path
        self.id2surface_form = {}
        self.surface_form2id = {}

    def get_id_2_surface_form_dict(self, allow_labels=False):
        d = {}
        for _id in self.id2surface_form:
            surface_form = self.get_from_wikidata_id(_id, allow_labels)
            if surface_form is not None:
                d[_id] = surface_form

        return d

    def load(self):
        print("Reading mapping from:", self.path)
        id2surface_form = {}
        with jsonlines.open(self.path) as f:
            for e in f:
                wikidata_id = e["wikidata_id"]
                info = e["information"]

                assert wikidata_id not in id2surface_form, "Duplicate Wikidata IDs"
                id2surface_form[wikidata_id] = info

        self.id2surface_form = id2surface_form
        self.construct_surface_form2id(verbose=False)

    def dump(self):
        print("Writing data_module to:", self.path)
        with jsonlines.open(self.path, "w") as writer:
            data = [{"wikidata_id": _id, "information": info_obj} for _id, info_obj in self.id2surface_form.items()]
            writer.write_all(data)

    def set_dict(self, id2surface_form_dict, simple_dict=True):
        if simple_dict:
            for key, value in id2surface_form_dict.items():
                id2surface_form_dict[key] = {"en_title": value}

        self.id2surface_form = id2surface_form_dict
        self.construct_surface_form2id()

    def construct_surface_form2id(self, verbose=True):
        surface_form2id = {}

        for _id, info_obj in self.id2surface_form.items():
            surface_form, provenance = self._get_surface_form_from_info_obj(info_obj)
            if surface_form in surface_form2id and verbose:
                print(
                    "Duplicate surface form for: \nWikidata ID {} --> {} and \nWikidata ID {} --> {}".format(
                        surface_form2id[surface_form],
                        self.id2surface_form[surface_form2id[surface_form]],
                        _id,
                        self.id2surface_form[_id],
                    )
                )

            surface_form2id[surface_form] = _id

        self.surface_form2id = surface_form2id

    @staticmethod
    def _get_surface_form_from_info_obj(info_obj):
        if "en_title" in info_obj:
            surface_form = info_obj["en_title"]
            provenance = "en_title"
        elif "en_label" in info_obj:
            surface_form = info_obj["en_label"]
            provenance = "en_label"
        else:
            raise Exception("Unexpected keys in info object:", info_obj)

        return surface_form, provenance

    @staticmethod
    def _query_wikidata_using_wikidata_id(wikidata_id, allow_labels=False, verbose=False):
        surface_form, provenance = None, None
        import pywikibot

        site = pywikibot.Site("wikidata", "wikidata")
        repo = site.data_repository()
        try:
            item = pywikibot.ItemPage(repo, "{}".format(wikidata_id))
            item.get(get_redirect=True)
            en_title = item.getSitelink("enwiki")
            surface_form = en_title
            provenance = "en_title"
        except pywikibot.exceptions.NoPageError as e:
            if allow_labels:
                try:
                    surface_form = item.get(get_redirect=True)["labels"]["en"]
                    provenance = "en_label"
                except Exception as e:
                    if verbose:
                        print("Label for item with id `{}` cannot be retrieved".format(wikidata_id))

        except pywikibot.exceptions.InvalidTitleError as e:
            if verbose:
                print("Wikidata item with id `{}` cannot be retrieved".format(wikidata_id))
            return None, None

        return surface_form, provenance

    @staticmethod
    def _construct_info_object(surface_form, provenance):
        return {provenance: surface_form}

    def update(self, key, value):
        id2surface_form = self.id2surface_form

        if key in id2surface_form:
            print(
                "Alert: the key `{}` is already present in the id2surface_form dict. \nObject {} will be overwriteen with {}".format(
                    key, id2surface_form[key], value
                )
            )

        id2surface_form[key] = value

        surface_form, _ = self._get_surface_form_from_info_obj(value)
        surface_form2id = self.surface_form2id
        if surface_form in surface_form2id:
            print(
                "Duplicate surface form for: \nWikidata ID {} --> {} and \nWikidata ID {} --> {}".format(
                    surface_form2id[surface_form],
                    self.id2surface_form[surface_form2id[surface_form]],
                    key,
                    self.id2surface_form[key],
                )
            )

        surface_form2id[surface_form] = key

    def get_from_wikidata_id(
        self, wikidata_id, return_provenance=False, allow_labels=False, query_wikidata=False, verbose=False
    ):
        id2surface_form = self.id2surface_form

        surface_form, provenance = None, None
        if wikidata_id not in id2surface_form:
            if query_wikidata:
                surface_form, provenance = self._query_wikidata_using_wikidata_id(
                    wikidata_id, allow_labels=allow_labels, verbose=verbose
                )

                if surface_form is not None:
                    info_object = self._construct_info_object(surface_form, provenance)
                    self.update(wikidata_id, info_object)

            if return_provenance:
                return surface_form, provenance
            else:
                return surface_form

        info_obj = id2surface_form[wikidata_id]

        surface_form, provenance = self._get_surface_form_from_info_obj(info_obj)

        if provenance == "en_label":
            if not allow_labels:
                if return_provenance:
                    return None, None
                else:
                    return None

        if return_provenance:
            return surface_form, provenance

        return surface_form
