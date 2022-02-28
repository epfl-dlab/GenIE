import matplotlib.pyplot as plt

# from genie.datamodule.utils.triplet_utils import TripletUtils
from genie.datamodule.utils import get_relations_and_relation_occurrence_counts_in_dataset
import numpy as np

from collections import Counter

import seaborn as sns
import math

from itertools import cycle

sns.set_theme(style="whitegrid")
sns.set_style("ticks")
sns.set_palette("deep")


class BucketPlotHelper(object):
    @staticmethod
    def get_bin_edges(_max, base=2):
        bin_edges = [0]

        for power in range(math.ceil(np.log(_max) / np.log(base)) + 1):
            bin_edges.append(base ** power)

        if bin_edges[-1] == _max:
            power += 1
            bin_edges.append(2 ** power)

        return bin_edges

    @staticmethod
    def get_bucket_label(b_id):
        if b_id > 0:
            return f"$2^{{{b_id - 1}}}$"

        return "None"

    @staticmethod
    def get_bucket_labels(b_ids):
        return [BucketPlotHelper.get_bucket_label(b_id) for b_id in b_ids]

    def get_bucket_ids(self, rel_names):
        return [self.get_bucket_id_for_rel_name(rel_name) for rel_name in rel_names]

    def get_bucket_id_for_rel_name(self, rel_name):
        return self.get_bucket_id_for_occ_count(self.rel2ref_occ[rel_name])

    def get_bucket_id_for_occ_count(self, value):
        assert value >= self.bin_edges[0]
        assert value < self.bin_edges[-1]

        for i in range(0, len(self.bin_edges) - 1):
            if value < self.bin_edges[i + 1]:
                return i

    def get_bucket_id2rels(self, rels):
        bucket_id2rels = {}

        for rel_name, bucket_id in zip(rels, self.get_bucket_ids(rels)):
            s = bucket_id2rels.get(bucket_id, set())
            s.add(rel_name)
            bucket_id2rels[bucket_id] = s

        return bucket_id2rels

    def __init__(self, reference_dataset):
        self.reference_dataset = reference_dataset
        _, self.rel2ref_occ = get_relations_and_relation_occurrence_counts_in_dataset(reference_dataset)
        self.rel_names_descending = [i[0] for i in sorted(list(self.rel2ref_occ.items()), key=lambda x: -x[1])]
        self.bin_edges = BucketPlotHelper.get_bin_edges(max(self.rel2ref_occ.values()))
        self.bucket_ids = list(range(len(self.bin_edges) - 1))
        self.bucket_id2num_ref_rels = Counter(self.get_bucket_ids(self.rel_names_descending))

    def plot_occurence_per_bucket_for_relations(self, rels):
        bucket_ids = self.get_bucket_ids(rels)
        num_rels_per_bucket_id = Counter(bucket_ids)

        fig, ax = plt.subplots(figsize=(14, 6))

        x = BucketPlotHelper.get_bucket_labels(self.bucket_ids)
        y = [num_rels_per_bucket_id[b_id] for b_id in self.bucket_ids]
        sns.barplot(x=x, y=y, palette=["b"] * len(x), label="Data distribution", hatch="//", edgecolor="black")
        ax.set_ylabel("Frequency")

        fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        plt.show()

        return num_rels_per_bucket_id

    def plot_twinx_barplot_with_train_dist(
        self,
        left_y,
        left_yaxis_label,
        bucket_id2rel_count_in_left=None,
        bucked_id2occ_count=None,
        bar_bucket_ids=None,
        figsize=(14, 6),
        ax1_hatch="\\\\",
        ax2_hatch="//",
        ax1_lim=(0, 1),
        ax2_lim=(None, None),
        save_to_file=None,
        alpha=1,
    ):
        left_yaxis_label = left_yaxis_label.capitalize()
        if bar_bucket_ids is None:
            x = BucketPlotHelper.get_bucket_labels(self.bucket_ids)
            bar_bucket_ids = self.bucket_ids
        else:
            x = BucketPlotHelper.get_bucket_labels(bar_bucket_ids)

        if bucket_id2rel_count_in_left is not None:
            x = [f"{label} \nR {bucket_id2rel_count_in_left.get(b_id, 0)}" for label, b_id in zip(x, bar_bucket_ids)]

        if bucked_id2occ_count is not None:
            x = [f"{label} \n{bucked_id2occ_count.get(b_id, 0)}" for label, b_id in zip(x, bar_bucket_ids)]

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("Relation occurrences in the training dataset")

        ### Left Plot
        plt.setp(ax.patches, linewidth=0)

        if isinstance(left_y, dict):
            left_y = {int(key): value for key, value in left_y.items()}
            # x = [f"{label} \n ({left_y.get(b_id, 0):.2f})" for label, b_id in zip(x, bar_bucket_ids)]
            if isinstance(list(left_y.values())[0], tuple) or isinstance(list(left_y.values())[0], list):
                yerr = [left_y.get(_id, (0, 0))[1] for _id in bar_bucket_ids]
                left_y = [left_y.get(_id, (0, 0))[0] for _id in bar_bucket_ids]
            else:
                left_y = [left_y.get(_id, 0) for _id in bar_bucket_ids]
                yerr = None

        sns.barplot(
            x=x,
            y=left_y,
            yerr=yerr,
            palette=["r"] * len(x),
            ax=ax,
            label=left_yaxis_label,
            hatch=ax1_hatch,
            edgecolor="black",
        )
        ax.set_ylabel(left_yaxis_label)
        ax.set_ylim(*ax1_lim)
        ###

        for error_bar in ax.lines:
            error_bar.set_zorder(10000)

        ### Right Plot: Data Distribution
        ax2 = ax.twinx()
        plt.setp(ax2.patches, linewidth=0)

        ax2_y = [self.bucket_id2num_ref_rels[b_id] for b_id in bar_bucket_ids]
        sns.barplot(
            x=x,
            y=ax2_y,
            palette=["b"] * len(x),
            ax=ax2,
            label="Data distribution",
            hatch=ax2_hatch,
            edgecolor="black",
            alpha=alpha,
        )
        ax2.set_ylabel("Frequency")
        ax2.set_ylim(*ax2_lim)
        ###

        ax.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, frameon=False)
        ax2.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax2.transAxes, frameon=False)
        plt.tight_layout(rect=(0, 0, 1, 0.98))

        if save_to_file is not None:
            plt.savefig(f"{save_to_file}.pdf", dpi=300)
            plt.savefig(f"{save_to_file}.png", dpi=300)

        # plt.show()

        return fig

    def plot_twinx_line_with_train_dist(
        self,
        left_ys,
        left_yaxis_label,
        bucket_id2rel_count_in_left=None,
        bucked_id2occ_count=None,
        bar_bucket_ids=None,
        figsize=(14, 6),
        ax1_hatch="\\\\",
        ax2_hatch="//",
        ax1_lim=(0, 1),
        ax2_lim=(None, None),
        save_to_file=None,
        alpha=1,
        capsize=0,
        linestyle="--",
        marker_style="o",
        marker_size=50,
        marker_linewidth=2.5,
        marker_color="r",
        show_plot=True,
        fig=None,
        ax=None,
        ax2=None,
        left_y_legends=None,
        drop_top_frame=True,
        legend_pos=None,
    ):
        c = cycle(["r", "GoldenRod", "violet", "yellow"])
        left_yaxis_label = left_yaxis_label.capitalize()
        if bar_bucket_ids is None:
            x = BucketPlotHelper.get_bucket_labels(self.bucket_ids)
            bar_bucket_ids = self.bucket_ids
        else:
            x = BucketPlotHelper.get_bucket_labels(bar_bucket_ids)

        if bucket_id2rel_count_in_left is not None:
            bucket_id2rel_count_in_left = {int(key): value for key, value in bucket_id2rel_count_in_left.items()}
            x = [f"{label} \nR {bucket_id2rel_count_in_left.get(b_id, 0)}" for label, b_id in zip(x, bar_bucket_ids)]

        if bucked_id2occ_count is not None:
            bucked_id2occ_count = {int(key): value for key, value in bucked_id2occ_count.items()}
            x = [f"{label} \n{bucked_id2occ_count.get(b_id, 0)}" for label, b_id in zip(x, bar_bucket_ids)]

        if ax is None:
            axes_given = False
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_xlabel("Number of relation occurrences in the training dataset")
        else:
            axes_given = True

        ### Left Plot
        plt.setp(ax.patches, linewidth=0)
        if not isinstance(left_ys, list):
            left_ys = [left_ys]

            if left_y_legends is None:
                left_y_legends = [left_yaxis_label]

            if not isinstance(left_y_legends, list):
                left_y_legends = [left_y_legends]

        for left_y, left_y_legend in zip(left_ys, left_y_legends):
            if isinstance(left_y, dict):
                left_y = {int(key): value for key, value in left_y.items()}
                # x = [f"{label} \n ({left_y.get(b_id, 0):.2f})" for label, b_id in zip(x, bar_bucket_ids)]
                if isinstance(list(left_y.values())[0], tuple) or isinstance(list(left_y.values())[0], list):
                    yerr = [left_y.get(_id, (0, 0))[1] for _id in bar_bucket_ids]
                    left_y = [left_y.get(_id, (0, 0))[0] for _id in bar_bucket_ids]
                else:
                    left_y = [left_y.get(_id, 0) for _id in bar_bucket_ids]
                    yerr = None

            color = next(c)
            sns.lineplot(x=x, y=left_y, ax=ax, color=color, label=left_y_legend, linestyle=linestyle)

            if yerr is not None:
                ax.errorbar(x=x, y=left_y, yerr=yerr, fmt="none", color="black", capsize=capsize, zorder=10000)

            if marker_size != 0:
                sns.scatterplot(
                    x=x,
                    y=left_y,
                    marker=marker_style,
                    color=color,
                    linewidth=marker_linewidth,
                    s=marker_size,
                    edgecolor="none",
                )

        if not axes_given:
            ax.set_ylabel(left_yaxis_label)
            ax.set_ylim(*ax1_lim)

        ## Right Plot: Data Distribution

        if not axes_given:
            ax2 = ax.twinx()
            plt.setp(ax2.patches, linewidth=0)

        ax2_y = [self.bucket_id2num_ref_rels[b_id] for b_id in bar_bucket_ids]
        sns.barplot(
            x=x,
            y=ax2_y,
            palette=["#004d99"] * len(x),
            ax=ax2,
            label="Distribution of relations",
            hatch=ax2_hatch,
            edgecolor="black",
            zorder=0,
            alpha=alpha,
        )

        if not axes_given:
            ax2.set_ylabel("Number of relations")
            ax2.set_ylim(*ax2_lim)
        ###

        if not axes_given:
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.set_frame_on(False)

        if drop_top_frame:
            ax.spines["top"].set_visible(False)
            ax2.spines["top"].set_visible(False)

        if legend_pos is None:
            legend_pos = (0.8, 1.03)

        ax.legend(loc="upper right", bbox_to_anchor=(1, 1.03), bbox_transform=ax.transAxes, frameon=False)
        ax2.legend(loc="upper right", bbox_to_anchor=legend_pos, bbox_transform=ax2.transAxes, frameon=False)
        plt.tight_layout(rect=(0, 0, 1, 0.98))

        if show_plot:
            plt.show()

        # # Only show ticks on the left and bottom spines
        # ax.yaxis.set_ticks_position('left')
        # ax.xaxis.set_ticks_position('bottom')

        if save_to_file is not None:
            plt.savefig(f"{save_to_file}.pdf", dpi=300)
            plt.savefig(f"{save_to_file}.png", dpi=300)

        return fig, ax, ax2


# ph = PlottingHelper(dataset)
# ph = PlottingHelper(dataset_nzs)
# ph.rc_train.head()
