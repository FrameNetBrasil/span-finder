from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from trankit import Pipeline

TRANKIT_PIPE = None


class Token:
    def __init__(self, text: str, idx: int):
        self.text = text
        self.idx = idx

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Token(text={self.text!r}, idx={self.idx})"


def spacy_tokenize(df):
    tokenizers = {
        "en": English().tokenizer,
        "pt": Portuguese().tokenizer,
        "es": Spanish().tokenizer,
    }

    unique_texts = df.drop_duplicates("text")[["text", "language"]]
    unique_texts["tokens"] = [
        [Token(text=token.text, idx=token.idx) for token in tokenizers[lang](text)]
        for lang, text in zip(unique_texts["language"], unique_texts["text"])
    ]

    return df.merge(unique_texts, on=["text", "language"], how="left")["tokens"]


def trankit_tokenize(df, gpu):
    trankit_languages = {"pt": "portuguese-gsd", "en": "english", "es": "spanish-gsd"}

    pipe = Pipeline("english", gpu=gpu)
    for language in trankit_languages.values():
        pipe.add(language)

    unique_texts = df.drop_duplicates("text")[["text", "language"]]

    text_to_tokens = {}
    for language, group in unique_texts.groupby("language"):
        pipe.set_active(trankit_languages[language])

        split_pos = (group["text"].str.len() + 5).cumsum().tolist()
        joint_string = ("\n" * 5).join(group["text"])
        tokens = pipe.tokenize(joint_string, is_sent=True)["tokens"]

        token_lists = [[] for _ in range(len(group))]
        sentence_idx = 0
        for token in tokens:
            if token["span"][0] >= split_pos[sentence_idx]:
                sentence_idx += 1
            token_sent_idx = token["span"][0] - (
                split_pos[sentence_idx - 1] if sentence_idx > 0 else 0
            )
            token_lists[sentence_idx].append(
                Token(text=token["text"], idx=token_sent_idx)
            )

        text_to_tokens.update(dict(zip(group["text"], token_lists)))

    unique_texts["tokens"] = unique_texts["text"].map(text_to_tokens)
    return df.merge(unique_texts, on=["text", "language"], how="left")["tokens"]


def get_token_spans(row):
    start_token = -1

    for i, token in enumerate(row["tokens"]):
        if token.idx >= row["startChar"] and start_token == -1:
            start_token = i
        if token.idx > row["endChar"]:
            end_token = i - 1
            break
    else:
        end_token = i

    return start_token, end_token
