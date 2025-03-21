import os
import logging
import pandas as pd
import xml.etree.ElementTree as ET

# FN 1.7 Namespace
NS = "{http://framenet.icsi.berkeley.edu}"


def get_frame_names(path):
    return {
        frame.get("ID"): frame.get("name")
        for frame in ET.parse(path).findall(f"{NS}frame")
    }


def get_lu_incorporations(release_path):
    files = os.listdir(os.path.join(release_path, "lu"))
    lu_incorporations = {}

    for filename in files:
        path = os.path.join(release_path, "lu", filename)
        root = ET.parse(path).getroot()

        fe = root.get("incorporatedFE")
        if fe is not None:
            lu_incorporations[root.get("ID")] = fe
    
    return lu_incorporations



def parse_file_spans(root, frame_names, lu_incorporations):
    spans = []

    for sentence in root.findall(f"{NS}sentence"):
        sent_id = sentence.get("ID")
        doc_id = sentence.get("docID")
        corpus_id = sentence.get("corpID")
        text = sentence.find(f"{NS}text").text

        for annoset in sentence.findall(f"{NS}annotationSet"):
            anno_id = annoset.get("ID")
            frame_id = annoset.get("frameID")
            lu_id = annoset.get("luID")
            target = annoset.find(
                f'{NS}layer[@name="Target"]/{NS}label[@name="Target"]'
            )

            if target is None or frame_id is None:
                continue

            frame = frame_names[frame_id]

            target_start = target.get("start")
            target_end = target.get("end")

            for fe in annoset.findall(f'{NS}layer[@name="FE"]/{NS}label'):
                fe_name = fe.get("name")
                fe_itype = fe.get("itype")
                fe_start = fe.get("start")
                fe_end = fe.get("end")

                if fe_itype is None or fe_itype == "INC":
                    spans.append(
                        [
                            corpus_id,
                            doc_id,
                            sent_id,
                            text,
                            anno_id,
                            False,
                            fe_itype or "NORMAL",
                            frame,
                            fe_name,
                            int(fe_start) if fe_start is not None else None,
                            int(fe_end) if fe_end is not None else None,
                            "en",
                        ]
                    )

            spans.append(
                [
                    corpus_id,
                    doc_id,
                    sent_id,
                    text,
                    anno_id,
                    True,
                    None,
                    frame,
                    lu_incorporations.get(lu_id, None),
                    int(target_start),
                    int(target_end),
                    "en",
                ]
            )

    return spans


def parse_spans(release_path, frames, fes):
    frame_names = get_frame_names(os.path.join(release_path, "frameIndex.xml"))
    lu_incorporations = get_lu_incorporations(release_path)

    files = os.listdir(os.path.join(release_path, "fulltext"))
    spans = []

    for filename in files:
        path = os.path.join(release_path, "fulltext", filename)
        root = ET.parse(path).getroot()
        spans.extend(parse_file_spans(root, frame_names, lu_incorporations))

    df = pd.DataFrame.from_records(
        spans,
        columns=[
            "idCorpus",
            "document",
            "idSentence",
            "text",
            "idAnnotationSet",
            "isTarget",
            "idInstantiationType",
            "frameName",
            "feName",
            "startChar",
            "endChar",
            "language",
        ],
    )

    # Manually update some data
    df.loc[
        (df["frameName"] == "Medical_intervention")
        & (df["feName"] == "Medical_condition"),
        "feName",
    ] = "Health_condition"
    df.loc[
        (df["frameName"] == "Medical_intervention")
        & (df["feName"] == "Medical_professional"),
        "feName",
    ] = "Healthcare_professional"

    # Filter out sentences in which one or more frames were not found on the structure
    df = df.merge(frames, how="left", on="frameName")
    not_found = df["idFrame"].isna()
    not_found_sents = df.loc[not_found, "text"].unique()
    if not_found.sum() > 1:
        logging.warning(
            "The following frames are annotated on 1.7 release but are not of the current FN structure: \n-"
            + "\n-".join(df.loc[not_found, "frameName"].unique())
        )
        logging.warning(
            f"{len(not_found_sents)} sentences with these frames annotated will be ignored"
        )
    df = df[~df["text"].isin(not_found_sents)]

    # Check for FE matching
    df = df.merge(fes, how="left", on=["idFrame", "feName"])
    not_found = ~df["isTarget"] & df["idFrameElement"].isna()

    if len(df[not_found]) > 0:
        distinct_pairs = df[not_found][["frameName", "feName"]].drop_duplicates()
        logging.warning(
            "Some (Frame, FE) pairs from 1.7 release could not be mapped to current DB: \n"
            + "\n".join(
                f"- Frame: {row['frameName']}, FE: {row['feName']}"
                for _, row in distinct_pairs.iterrows()
            )
            + "\n\n They could've been renamed or the DB must be revised."
        )

        # We'll remove it to prevent errors
        df = df[~not_found]

    df["idFrame"] = df["idFrame"].astype("Int64")
    df["idFrameElement"] = df["idFrameElement"].astype("Int64")

    return df
