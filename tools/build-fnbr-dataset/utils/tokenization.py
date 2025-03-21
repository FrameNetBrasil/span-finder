from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from spacy.attrs import ORTH

EN_TOKENIZER = English().tokenizer
ES_TOKENIZER = Spanish().tokenizer
PT_TOKENIZER = Portuguese().tokenizer

PT_TOKENIZER.add_special_case("[NUM]", [{ ORTH: "[NUM]"}])
PT_TOKENIZER.add_special_case("[NAME]", [{ ORTH: "[NAME]"}])
PT_TOKENIZER.add_special_case("[DATE]", [{ ORTH: "[DATE]"}])
PT_TOKENIZER.add_special_case("[TIME]", [{ ORTH: "[TIME]"}])
PT_TOKENIZER.add_special_case("[VENUE]", [{ ORTH: "[VENUE]"}])
PT_TOKENIZER.add_special_case("[TIMESPAN]", [{ ORTH: "[TIMESPAN]"}])
PT_TOKENIZER.add_special_case("[...]", [{ ORTH: "[...]"}])

def spacy_tokenize(row):
    '''
    Tokenization will be done using spaCy models.
    A different model is used depending on whether the text is in English or in Portuguese.
    '''
    tokenizer = PT_TOKENIZER if row["language"] == "pt" else EN_TOKENIZER
    return list(tokenizer(row["text"]))


def spacy_linearize(tokens):
    '''
    Transforms a list of spaCy tokens back to string.
    '''
    return ''.join(t.text_with_ws for t in tokens)


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
