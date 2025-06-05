import sys
import json
from argparse import ArgumentParser
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese
from sftp import SpanPredictor

# Language-specific tokenizers
tokenizers = {
    "en": English().tokenizer,
    "pt": Portuguese().tokenizer,
    "es": Spanish().tokenizer,
}


def tokenize(sentences, tokenizer):
    tokens, spans = [], []
    for sent in sentences:
        if not isinstance(sent, str) or sent.strip() == "":
            continue
        token_list = []
        span_list = []
        for token in tokenizer(sent):
            if token.text.strip() != "":
                token_list.append(token.text)
                span_list.append((token.idx, token.idx + len(token)))
        tokens.append(token_list)
        spans.append(span_list)
    return tokens, spans


def include_char_spans(annotations, char_spans):
    for annotation in annotations:
        start_char = char_spans[annotation["span"][0]][0]
        end_char = char_spans[annotation["span"][1]][1]
        annotation["char_span"] = [start_char, end_char]
        if "children" in annotation:
            include_char_spans(annotation["children"], char_spans)


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', metavar='MODEL_PATH', default="/srv/model", type=str, help="Path to the model")
    parser.add_argument('-d', metavar='DEVICE', type=int, default=-1, help="CUDA device (-1 for CPU)")
    parser.add_argument('-l', metavar='LANG', default='en', type=str, help="Language code: en, pt, es")
    args = parser.parse_args()

    if args.l not in tokenizers:
        raise ValueError(f"Invalid language code '{args.l}'. Must be one of: {list(tokenizers.keys())}")

    sentences = [line.strip() for line in sys.stdin if line.strip()]
    tokens, char_spans = tokenize(sentences, tokenizers[args.l])

    predictor = SpanPredictor.from_path(args.m, cuda_device=args.d)
    model_outputs = predictor.predict_batch_sentences(tokens, max_tokens=512)

    for i, output in enumerate(model_outputs):
        sent_tokens = output.sentence
        annotations = output.span.to_json()["children"]
        sent_char_spans = char_spans[i]
        include_char_spans(annotations, sent_char_spans)
        json.dump({
            'tokens': sent_tokens,
            'annotations': annotations
        }, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
