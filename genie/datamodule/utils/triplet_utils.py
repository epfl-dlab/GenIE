import re


class TripletUtils(object):
    @staticmethod
    def convert_text_sequence_to_text_triples(text, verbose=True, return_set=True):
        text_parts = [element.strip() for element in re.split(r"<sub>|<rel>|<obj>|<et>", text) if element.strip()]
        if verbose and len(text_parts) % 3 != 0:
            print(f"Textual sequence: ```{text}``` does not follow the <sub>, <rel>, <obj>, <et> format!")

        text_triples = [tuple(text_parts[i : i + 3]) for i in range(0, len(text_parts) - 2, 3)]

        if not return_set:
            return text_triples

        unique_text_triples = set(text_triples)

        if verbose and len(unique_text_triples) != len(text_triples):
            print(f"Textual sequence: ```{text}``` has duplicated triplets!")

        return unique_text_triples

    @staticmethod
    def triples_to_output_format(triples):
        output_triples = []

        for t in triples:
            sub, rel, obj = t
            formatted_triple = "{} {}{} {}{} {}{}".format(
                " <sub>", sub.strip(), " <rel>", rel.strip(), " <obj>", obj.strip(), " <et>"
            )
            output_triples.append(formatted_triple)

        output = "".join(output_triples)
        return output

    @staticmethod
    def process_triple_of_ids(triple, ent_mapping, rel_mapping, query_wikidata, allow_labels):
        """returns match_status, id form, surface form, provenance """
        if len(triple) != 3:
            raise Exception("Invalid triple:", triple)

        head_id, rel_id, tail_id = triple

        head_s, head_p = ent_mapping.get_from_wikidata_id(
            head_id, return_provenance=True, query_wikidata=query_wikidata, allow_labels=allow_labels
        )
        tail_s, tail_p = ent_mapping.get_from_wikidata_id(
            tail_id, return_provenance=True, query_wikidata=query_wikidata, allow_labels=allow_labels
        )

        rel_s, rel_p = rel_mapping.get_from_wikidata_id(
            rel_id, return_provenance=True, query_wikidata=query_wikidata, allow_labels=allow_labels
        )

        surface_form = [head_s, rel_s, tail_s]
        provenance = [head_p, rel_p, tail_p]

        if head_p is None or rel_p is None or tail_p is None:
            status = "no_match"
        elif head_p == "en_label" or rel_p == "en_label" or tail_p == "en_label":
            status = "label"
        elif head_p == "en_title" and rel_p == "en_title" and tail_p == "en_title":
            status = "title"
        else:
            raise Exception("Invalid provenance")

        return status, triple, surface_form, provenance
