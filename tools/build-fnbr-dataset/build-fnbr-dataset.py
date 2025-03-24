import time

import os
import argparse
import json
import logging
import tomllib
import numpy as np
import pandas as pd
from operator import itemgetter
from utils.ehr_post import deanonymize
from utils.fn17_parser import parse_spans
from utils.fnbr_db import get_fn_structure, get_spans
from utils.tokenization import get_token_spans, spacy_tokenize, trankit_tokenize

# Constants
VALID_SOURCES = {"fn17", "fnbr", "ehr"}

# Logging Configuration
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def test_spans(spans):
    sample = spans[(spans["idInstantiationType"] != "INC")].sample(n=min(len(spans), 15))

    for _, row in sample.iterrows():
        index = row.name
        char_based = row["text"][int(row["startChar"]):int(row["endChar"]) + 1]
        token_based = ' '.join(map(str,row["tokens"][int(row["startToken"]):int(row["endToken"]) + 1]))
        print(f"{index} - {char_based}      {token_based}")


# Data Processing Functions
def build_instances(df):
    instances = []
    for _, sent_group in df.groupby("text"):
        annotations = []

        for annoset_id, annoset in sent_group.groupby("idAnnotationSet"):
            children = []
            target_idx = annoset["isTarget"]
            target = annoset[target_idx].iloc[0]
            target_span = [int(target["startToken"]), int(target["endToken"])]

            if annoset[~target_idx]["idFrameElement"].isna().sum() > 0:
                logging.error(
                    f"Found instance with more than one target: {annoset_id} (idAnnotationSet)"
                )

            # Incorporations
            incorporation_idx = target_idx | (annoset["idInstantiationType"] == "INC")
            incorporations = annoset.loc[incorporation_idx, "idFrameElement"].dropna().unique()
            for fe in incorporations:
                children.append(
                    {"span": target_span, "label": f"fe_{fe}", "children": []}
                )

            # Normal instantiation
            for _, fe in annoset[~incorporation_idx].iterrows():
                children.append(
                    {
                        "span": [int(fe["startToken"]), int(fe["endToken"])],
                        "label": f"fe_{fe['idFrameElement']}",
                        "children": [],
                    }
                )

            annotations.append(
                {
                    "span": [int(target["startToken"]), int(target["endToken"])],
                    "label": f"frm_{target['idFrame']}",
                    "children": children,
                }
            )

        instances.append(
            {
                "tokens": list(map(str, sent_group.iloc[0]["tokens"])),
                "annotations": annotations,
                "meta": {
                    "corpus": int(sent_group.iloc[0]["idCorpus"]),
                    "language": sent_group.iloc[0]["language"],
                },
            }
        )
    return instances


def split_instances(instances, percentages):
    splits = (np.cumsum(percentages) * len(instances) / 100).astype(int)
    np.random.shuffle(instances)
    train, validate, test = np.split(instances, splits[:-1])

    return train, validate, test


def write_ontology(frames, fes, spans, filename):
    lines = list()

    # Build ontology out of annotated entities only
    frames = frames[frames["idFrame"].isin(spans["idFrame"].unique())]
    fes = fes[fes["idFrame"].isin(spans["idFrameElement"].unique())]

    lines.append(
        "@@VIRTUAL_ROOT@@\t" + "\t".join(frames["idFrame"].apply(lambda i: f"frm_{i}"))
    )
    for frame, group in fes.groupby("idFrame")["idFrameElement"]:
        lines.append(f"frm_{frame}\t" + "\t".join(group.map(lambda s: f"fe_{s}")))

    with open(filename, "w") as fp:
        fp.write("\n".join(lines))


def write_jsonl(data, filename):
    with open(filename, "w") as fp:
        fp.writelines((map(lambda d: json.dumps(d) + "\n", data)))


# Argument Parsing Functions
def validate_sources_arg(args, parser):
    sources = args.sources.split(",")

    invalid_sources = set(sources) - VALID_SOURCES
    if invalid_sources:
        parser.error(
            f"Invalid sources provided: {', '.join(invalid_sources)}. "
            f"Valid options are: {', '.join(VALID_SOURCES)}"
        )

    if "fn17" in sources and args.fn17_path is None:
        parser.error(
            "source 'fn17': The path for the FrameNet 1.7 release must be informed (fn17_path)"
        )

    if "ehr" in sources and args.ehr_originaldb_path is None:
        parser.error(
            "source 'ehr': The path for the original sententeces file must be informed (ehr_originaldb_path)"
        )


def validate_splits_arg(args, parser):
    splits = args.splits.split(",")

    if len(splits) != 3:
        parser.error("'splits': must specify exactly 3 numbers")

    if sum(map(int, splits)) != 100:
        parser.error("'splits': not adding to 100")


def validate_structure_db_arg(args, parser):
    if args.structure_db not in args.db_config:
        parser.error(
            f"'structuredb': value {args.structure_db} not found in .toml config file"
        )


def validate_ehr_originaldb_path_arg(args, parser):
    try:
        df = pd.read_csv(args.ehr_originaldb_path, nrows=1)

        if "text" not in df.columns:
            parser.error(
                "'ehr_originaldb_path': CSV must have a 'text' column for anonymized sentences"
            )
        if "original" not in df.columns:
            parser.error(
                "'ehr_originaldb_path': CSV must have a 'original' column for pre-anonymization sentences"
            )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
        parser.error("'ehr_originaldb_path': must be a valid CSV file")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a training dataset in .jsonl format from specified sources."
    )

    parser.add_argument(
        "output_folder", type=str, help="Path to a folder to save the .jsonl dataset."
    )
    parser.add_argument(
        "--db_config",
        type=str,
        default="./config.toml",
        help="Path to the database connection configuration file (.toml)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        required=True,
        help="Specify sources as a comma-separated list (e.g., fn17,ehr). Valid values: fn17, fnbr, ehr",
    )
    parser.add_argument(
        "--fn17_path",
        type=str,
        help="Path of the folder containing the FrameNet 1.7 release",
    )
    parser.add_argument(
        "--ehr_originaldb_path",
        type=str,
        help="Path of the file containing the original sentences before anonymization. Must be CSV with columns 'text' and 'original'",
    )
    parser.add_argument(
        "--structure_db",
        type=str,
        default="ehr_db",
        help="The DB from which the FrameNet structure will be extracted from. Must be one of the databases on the db_config",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="trankit",
        choices=["trankit", "spacy"],
        help="The tokenizer to be used",
    )
    parser.add_argument(
        "--use_gpu",
        type=str,
        default=True,
        help="Whether the GPU will be used for Trankit tokenizer",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="70,15,15",
        help="The percentage of the training, validation and test splits of the dataset. Comma-separated.",
    )

    args = parser.parse_args()

    with open(args.db_config, "rb") as fp:
        args.db_config = tomllib.load(fp)

    validate_sources_arg(args, parser)
    validate_splits_arg(args, parser)
    validate_structure_db_arg(args, parser)
    validate_ehr_originaldb_path_arg(args, parser)

    args.sources = args.sources.split(",")
    args.splits = list(map(int, args.splits.split(",")))

    return args


# Main Execution Block
if __name__ == "__main__":
    args = parse_args()
    logging.info("Starting dataset preparation...")

    logging.info(
        f"The base FrameNet structure is the one from the following DB config: {args.structure_db}"
    )
    frames, fes = get_fn_structure(args.db_config[args.structure_db])
    df = pd.DataFrame()

    if "fn17" in args.sources:
        logging.info("Parsing FrameNet 1.7 data...")
        df = pd.concat(
            [df, parse_spans(args.fn17_path, frames, fes)], ignore_index=True
        )

    if "fnbr" in args.sources:
        logging.info("Loading FrameNet Brasil annotation...")
        df = pd.concat(
            [df, get_spans(args.db_config, "fnbr_db", frames, fes)], ignore_index=True
        )

    if "ehr" in args.sources:
        logging.info("Loading EHR annotation...")
        spans = get_spans(args.db_config, "ehr_db", frames, fes)
        logging.info("Deanonymizing...")
        spans = deanonymize(spans, args.ehr_originaldb_path)
        df = pd.concat([df, spans], ignore_index=True)

    logging.info("Tokenizing data...")
    if args.tokenizer == "trankit":
        logging.info("Using trankit tokenizer...")
        start = time.perf_counter()
        df["tokens"] = trankit_tokenize(df, args.use_gpu)
        end = time.perf_counter()
        logging.info(f"tokenization took {end - start:.4f} seconds")
    else:
        logging.info("Using spaCy tokenizer...")
        start = time.perf_counter()
        df["tokens"] = spacy_tokenize(df, args.use_gpu)
        end = time.perf_counter()
        logging.info(f"tokenization took {end - start:.4f} seconds")

    logging.info("Calculating token spans...")
    token_spans = df.apply(get_token_spans, axis=1)
    df["startToken"] = token_spans.map(itemgetter(0))
    df["endToken"] = token_spans.map(itemgetter(1))

    logging.info("Building instances...")
    instances = build_instances(df)

    logging.info("Splitting dataset into train, validation, and test sets...")
    train, validate, test = split_instances(instances, args.splits)

    logging.info("Writing ontology file...")
    write_ontology(frames, fes, df, os.path.join(args.output_folder, "ontology"))

    logging.info("Writing dataset to .jsonl files...")
    write_jsonl(train, os.path.join(args.output_folder, "train.jsonl"))
    write_jsonl(validate, os.path.join(args.output_folder, "val.jsonl"))
    write_jsonl(test, os.path.join(args.output_folder, "test.jsonl"))

    logging.info("Dataset preparation completed successfully!")
