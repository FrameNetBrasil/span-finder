import logging
import pandas as pd
from sqlalchemy import create_engine


def get_fn_structure(config):
    engine = create_engine(
        f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    )

    frames = pd.read_sql(
        """
        select
            frm1.idFrame,
            trim(frm1.name) as 'frameName_pt',
            trim(frm2.name) as 'frameName_en'
        from view_frame frm1
        join view_frame frm2 on frm2.idFrame = frm1.idFrame
        where frm1.idLanguage = 1 and frm2.idLanguage = 2;""",
        engine,
    )

    fes = pd.read_sql(
        """
        select
            fe1.idFrame,
            fe1.idFrameElement,
            trim(fe1.name) as 'feName_pt',
            trim(fe2.name) as 'feName_en',
            fe1.coreType
        from view_frameelement fe1
        join view_frameelement fe2 on fe2.idFrameElement = fe1.idFrameElement
        where fe1.idLanguage = 1 and fe2.idLanguage = 2;
        """,
        engine,
    )

    lus = pd.read_sql(
        """
        select distinct
            lu.idFrame,
            lu.idLU,
            lu.incorporatedFE as 'idFrameElement',
            trim(lu.name),
            lu.idLanguage as 'language'
        from view_lu lu
        where lu.idLanguage in (1, 2, 3);""",
        engine,
    )

    lus["idFrameElement"] = lus["idFrameElement"].replace({0: None}).astype("Int64")
    lus["language"] = lus["language"].replace({1: "pt", 2: "en", 3: "es"})

    return (
        frames,
        fes[fes["idFrame"].isin(frames["idFrame"])],
        lus[lus["idFrame"].isin(frames["idFrame"])],
    )


def get_spans_40(engine, corpora, exclude_docs):
    query = """
        --
        -- Targets
        --
        select
            c.idCorpus,
            d.name as 'document',
            s.idSentence,
            s.text,
            a.idAnnotationSet,
            true as 'isTarget',
            null as 'idInstantiationType',
            f.name as 'frameName_en',
            fe.defaultName as 'feName_en',
            gl.startChar,
            gl.endChar,
            s.idLanguage as 'language',
            f.idFrame,
            l.incorporatedFE as 'idFrameElement'
        from view_annotation_text_gl gl
        join view_annotationset a on a.idAnnotationSet = gl.idAnnotationSet
        join view_document d on d.idDocument = a.idDocument
        join corpus c on c.idCorpus = d.idCorpus
        join sentence s on s.idSentence = a.idSentence
        join view_lu l on l.idLu = a.idLu
        join view_frame f on f.idFrame = l.idFrame
        left join frameelement fe on fe.idFrameElement = l.incorporatedFE
        where c.entry in %(corpora)s
            and d.name not in %(exclude_docs)s
            and d.idLanguage = 2
            and f.idLanguage = 2
            and s.idLanguage in (1, 2, 3)
        union all
        --
        -- FEs (normal and incorporation)
        --
        select 
            c.idCorpus,
            d.name as 'document',
            s.idSentence,
            s.text,
            a.idAnnotationSet,
            false as 'isTarget',
            fe.idInstantiationType,
            f.name as 'frameName_en',
            fe.name as 'feName_en',
            fe.startChar,
            fe.endChar,
            s.idLanguage,
            fe.idFrame,
            fe.idFrameElement
        from view_annotation_text_fe fe
        join view_annotationset a on a.idAnnotationSet = fe.idAnnotationSet
        join view_document d on d.idDocument = a.idDocument
        join corpus c on c.idCorpus = d.idCorpus
        join sentence s on s.idSentence = a.idSentence
        join view_frame f on f.idFrame = fe.idFrame
        where c.entry in %(corpora)s
            and d.name not in %(exclude_docs)s
            and fe.idLanguage = 2
            and d.idLanguage = 2
            and f.idLanguage = 2
            and s.idLanguage in (1, 2, 3)
            and fe.idInstantiationType in (12, 17);"""

    params = {"corpora": tuple(corpora), "exclude_docs": tuple(exclude_docs or [""])}
    df = pd.read_sql(query, engine, params=params)

    df["isTarget"] = df["isTarget"].astype(bool)
    df["idInstantiationType"] = df["idInstantiationType"].replace(
        {12: "NORMAL", 17: "INC"}
    )
    df["language"] = df["language"].replace({1: "pt", 2: "en", 3: "es"})
    df["frameName_en"] = df["frameName_en"].str.strip()
    df["feName_en"] = df["feName_en"].str.strip()

    return df


def get_spans_38(engine, corpora, exclude_docs):
    query = """
        select
            corpus.idCorpus,
            sentence.`text`,
            annotationset.idAnnotationSet,
            case when layer.idLayerType = 2 then true else false end as 'isTarget',
            frmentry.name as 'frameName_en',
            feentry.name as 'feName_en',
            label.startChar,
            label.endChar,
            sentence.idLanguage as 'language',
            frame.idFrame,
            coalesce(frameelement.idFrameElement, lu.incorporatedFE) as 'idFrameElement'
        from corpus
        join document on document.idCorpus = corpus.idCorpus 
        join sentence on sentence.idDocument = document.idDocument
        join annotationset on annotationset.idSentence = sentence.idSentence
        join layer on layer.idAnnotationSet = annotationset.idAnnotationSet
        join label on label.idLayer = layer.idLayer
        join lu on lu.`idEntity` = annotationset.idEntityRelated 
        join frame on frame.idFrame = lu.idFrame
        left join entry frmentry on frmentry.entry = frame.entry and frmentry.idLanguage = 2
        left join frameelement on frameelement.`idEntity` = label.idLabelType
        left join entry feentry on feentry.entry = frameelement.entry and feentry.idLanguage = 2
        where sentence.idLanguage in (1,2,3)
            and corpus.entry in %(corpora)s
            and document.entry not in %(exclude_docs)s
            and layer.idLayerType in (1, 2) -- FE and target
            and label.idInstantiationType in (12, 17) -- normal instantiation and incorporation"""

    params = {"corpora": tuple(corpora), "exclude_docs": tuple(exclude_docs or [""])}
    df = pd.read_sql(query, engine, params=params)

    df["isTarget"] = df["isTarget"].astype(bool)
    df["language"] = df["language"].replace({1: "pt", 2: "en", 3: "es"})

    return df


def get_spans(all_config, config_key, frames, fes):
    config = all_config[config_key]
    engine = create_engine(
        f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    )

    if config["version"] == "4.0":
        df = get_spans_40(engine, config["corpora"], config.get("exclude_docs", None))
    elif config["version"] == "3.8":
        df = get_spans_38(engine, config["corpora"], config.get("exclude_docs", None))
    else:
        raise ValueError(
            'Invalid value for DB \'version\'. Must be one of: "4.0", "3.8".'
        )

    # Filter out sentences in which one or more frames were not found on the structure
    df = df.merge(frames, how="left", on="frameName_en")
    not_found = df["idFrame_y"].isna()
    not_found_sents = df.loc[not_found, "text"].unique()
    if not_found.sum() > 1:
        logging.warning(
            f"The following frames are annotated on {config_key} but are not of the current FN structure: \n-"
            + "\n-".join(df.loc[not_found, "frameName_en"].unique())
        )
        logging.warning(
            f"{len(not_found_sents)} sentences where these frames annotated will be ignored"
        )
    df = df[~df["text"].isin(not_found_sents)]
    df["idFrame"] = df["idFrame_y"]
    df.drop(columns=["idFrame_x", "idFrame_y"], inplace=True)

    # Check for FE matching
    df = df.merge(fes, how="left", on=["idFrame", "feName_en"])
    not_found = ~df["isTarget"] & df["idFrameElement_y"].isna()

    if len(df[not_found]) > 0:
        distinct_pairs = df[not_found][["frameName_en", "feName_en"]].drop_duplicates()
        logging.warning(
            f"Some (Frame, FE) pairs from {config_key} could not be mapped to current structure: \n"
            + "\n".join(
                f"- Frame: {row['frameName_en']}, FE: {row['feName_en']}"
                for _, row in distinct_pairs.iterrows()
            )
            + "\n\n They could've been renamed or the DB must be revised."
        )

        # We'll remove it to prevent errors
        df = df[~not_found]

    df["idFrame"] = df["idFrame"].astype("Int64")
    df["idFrameElement"] = df["idFrameElement_y"].astype("Int64")
    df.drop(columns=["idFrameElement_x", "idFrameElement_y"], inplace=True)

    return df
