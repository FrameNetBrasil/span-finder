import pandas as pd
import diff_match_patch as dmp_module
from thefuzz import fuzz, process

dmp = dmp_module.diff_match_patch()


def get_text_diff_offsets_by_position(source, target, diff=None):
    if not diff:
        diff = dmp.diff_main(source, target)

    offset = -1
    breakpoints = []
    for op, text in diff:
        if op != 1:  # means that text is already in source
            offset += len(text)

        if op != 0:  # text is exclusive to source or target
            breakpoints.append((offset, len(text) * op))

    return breakpoints


def get_adjusted_position(position, diffs):
    i = 0
    diff = 0
    while i < len(diffs) and position >= diffs[i][0]:
        diff += diffs[i][1]
        i += 1
    return position + diff


def deanonymize(spans, originaldb_path):
    df_orig = pd.read_csv(originaldb_path)
    orig_dict = {k: v for k, v in zip(df_orig["text"], df_orig["original"])}
    all_sents = spans["text"].unique()

    text_to_orig = {}
    text_to_diffs = {}
    for sent in all_sents:
        if sent in orig_dict:
            text_to_orig[sent] = orig_dict[sent].strip()
        else:
            result = process.extractOne(sent, df_orig["text"], scorer=fuzz.ratio)
            text_to_orig[sent] = orig_dict[result].strip()

        text_to_diffs[sent] = get_text_diff_offsets_by_position(
            sent, text_to_orig[sent]
        )

    spans["diff"] = spans["text"].map(text_to_diffs)
    spans["text"] = spans["text"].map(text_to_orig)
    spans["startChar"] = spans.apply(
        lambda r: get_adjusted_position(r["startChar"], r["diff"]), axis=1
    )
    spans["endChar"] = spans.apply(
        lambda r: get_adjusted_position(r["endChar"], r["diff"]), axis=1
    )

    spans.drop(columns=["diff"], inplace=True)

    return spans
