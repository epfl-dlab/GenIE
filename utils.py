import os
import pickle
from collections import Counter

import jsonlines
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def count_matches_kilt_jsonl(input_file_path):
    print("============Counting matches for: {}============".format(input_file_path))
    with jsonlines.open(input_file_path) as f:
        processed_data = [e for e in f]

    item_prov = []
    triplet_prov = []
    instance_prov = []
    for instance in processed_data:
        prov_triplets = instance["output"][0]["non_formatted_surface_output_provenance"]
        for triplet in prov_triplets:
            for item in triplet:
                item_prov.append(item)

        triplet_status = instance["output"][0]["non_formatted_triples_match_status"]
        triplet_prov.extend(triplet_status)
        instance_prov.append(instance["instance_matching_status"])

    print("===Instance level counter==")
    print("The dataset consists of {} samples".format(len(processed_data)))
    print(Counter(instance_prov))

    print("===Triplet level counter==")
    print(Counter(triplet_prov))

    print("===Item level counter===")
    print(Counter(item_prov))


def get_duplicate_values(d):
    seen = set()
    dup_values = set()

    for key, value in d.items():
        if value not in seen:
            seen.add(value)
        else:
            dup_values.add(value)

    dup_keys = set()
    dup_details = {}

    for key, value in d.items():
        if value in dup_values:
            dup_keys.add(key)

            temp = dup_details.get(value, [])
            temp.append(key)
            dup_details[value] = temp

    return dup_keys, dup_details


def get_entity_dict_file(processed_wikidata_dicts_folder):
    return os.path.join(processed_wikidata_dicts_folder, "wikidata_entity_id_title2data")


def get_relation_dict_file(processed_wikidata_dicts_folder):
    return os.path.join(processed_wikidata_dicts_folder, "wikidata_relation_id_label2data")


def read_and_process_entity_dict(entity_output_file, verbose=False):
    with open(entity_output_file, "rb") as f:
        entity_id_title2data = pickle.load(f)

    entity_id2title = {_id: title for _id, title in entity_id_title2data.keys()}
    duplicates, dup_details = get_duplicate_values(entity_id2title)
    if verbose:
        print("Number of entity entries affected by non-unique titles:", len(duplicates))
        print("Number of titles duplicated:", len(dup_details))
    for key in duplicates:
        del entity_id2title[key]
    title2entity_id = {title: _id for _id, title in entity_id2title.items()}
    assert len(entity_id2title) == len(title2entity_id)  # No duplicates
    if verbose:
        print("Number of entity ids paired with unique titles:", len(entity_id2title))
    return entity_id2title, title2entity_id


def read_and_process_relations_dict(relation_output_file, verbose=False):
    with open(relation_output_file, "rb") as f:
        relation_id_label2data = pickle.load(f)

    relation_id2label = {_id: label for _id, label in relation_id_label2data.keys()}
    duplicates, dup_details = get_duplicate_values(relation_id2label)
    if verbose:
        print("Number of relation entries affected by non-unique labels:", len(duplicates))
        print("Number of labels duplicated:", len(dup_details))
    for key in duplicates:
        del relation_id2label[key]
    label2relation_id = {label: _id for _id, label in relation_id2label.items()}
    assert len(relation_id2label) == len(label2relation_id)  # No duplicates
    if verbose:
        print("Number of relation ids paired with unique labels:", len(label2relation_id))
    return relation_id2label, label2relation_id


class Plotting:
    @staticmethod
    def get_subplots(num_sublots, fig_size=None, sharex=False, sharey=False, fig_title=None):
        if fig_size is None:
            fig, axs = plt.subplots(num_sublots[0], num_sublots[1], sharex=sharex, sharey=sharey)
        else:
            fig, axs = plt.subplots(
                num_sublots[0],
                num_sublots[1],
                figsize=fig_size,
                sharex=sharex,
                sharey=sharey,
            )

        if fig_title is not None:
            fig.suptitle(fig_title)

        return fig, axs

    @staticmethod
    def remove_top_and_right_lines(ax):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    @staticmethod
    def plot_with_hist_on_x_single(x, y_label_pairs, x_all_vals, x_label, title, file_name=None):
        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(5, 4)  # first one is vertical

        axs = [fig.add_subplot(gs[1:, :])]
        eps = 0.05
        axs[0].set_ylim([-eps, 1 + eps])
        # utils.Plotting.remove_top_and_right_lines(axs[0])

        axs_marginals = [fig.add_subplot(gs[0, :], sharex=axs[0])]
        Plotting.remove_top_and_right_lines(axs_marginals[0])
        plt.setp(axs_marginals[0].get_xticklabels(), visible=False)

        # fig, axs = plt.subplots(1, 2, sharey=True, )
        for y, label in y_label_pairs:
            axs[0].plot(x, y, label=label)

        axs[0].set_xlabel(x_label)
        axs[0].set_title(title)
        ###

        for ax in axs_marginals:
            ax.hist(x_all_vals, bins="auto")

        axs[0].legend()

        if file_name is not None:
            plt.tight_layout(rect=(0, 0, 1, 0.98))
            fig.savefig(os.path.join("plots", f"{file_name}.png"))
            fig.savefig(os.path.join("plots", f"{file_name}.pdf"))

        plt.show()

    @staticmethod
    def plot_with_hist_on_x_double(x, y_label_pairs, x_all_vals, x_labels, titles, file_name=None):
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(5, 4)  # first one is vertical

        axs = [fig.add_subplot(gs[1:, 0:2])]
        eps = 0.05
        axs[0].set_ylim([-eps, 1 + eps])
        axs.append(fig.add_subplot(gs[1:, 2:], sharey=axs[0]))

        axs_marginals = [fig.add_subplot(gs[0, 0:2], sharex=axs[0])]
        axs_marginals.append(fig.add_subplot(gs[0, 2:], sharex=axs[1], sharey=axs_marginals[0]))
        for i, ax in enumerate(axs_marginals):
            plt.setp(ax.get_xticklabels(), visible=False)
            if i != 0:
                plt.setp(ax.get_yticklabels(), visible=False)

        for i in range(len(y_label_pairs)):
            y, label = y_label_pairs[i]
            axs[i].plot(x, y, label=label)
            axs[i].set_xlabel(x_labels[i])
            axs[i].set_title(titles[i])

        for ax in axs_marginals:
            ax.hist(x_all_vals, bins="auto")

        axs[0].legend()
        axs[1].legend()

        if file_name is not None:
            plt.tight_layout(rect=(0, 0, 1, 0.98))
            fig.savefig(os.path.join("plots", f"{file_name}.png"))
            fig.savefig(os.path.join("plots", f"{file_name}.pdf"))

        plt.show()
