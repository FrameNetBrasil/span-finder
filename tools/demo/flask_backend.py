import json
from argparse import ArgumentParser
from typing import *

from flask import Flask
from flask import request
from flask import jsonify

from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.pt import Portuguese

from sftp import SpanPredictor, Span

parser = ArgumentParser()
parser.add_argument('-m', metavar='MODEL_PATH', default="/srv/model", type=str)
parser.add_argument('-p', metavar='PORT', type=int, default=7749)
parser.add_argument('-d', metavar='DEVICE', type=int, default=-1)
args = parser.parse_args()

with open(args.m + '/config.json') as fp:
    m_config = json.load(fp)
with open(args.m + '/dataset_meta.json') as fp:
    ds_meta = json.load(fp)
    names = ds_meta["names"]

assert m_config["dataset_reader"]["dataset_timestamp"] == ds_meta["timestamp"]

with open('tools/demo/flask_template.html') as fp:
    template = fp.read()

tokenizers = { "en": English().tokenizer, "pt": Portuguese().tokenizer, "es": Spanish().tokenizer,}
predictor = SpanPredictor.from_path(args.m, cuda_device=args.d)
app = Flask(__name__)
default_sentence = '因为 आरजू です vegan , هي купил soja .'


def tokenize(sentences, lang="en"):
    tokens, spans = [], []
    for sent in sentences:
        if not isinstance(sent, str) or sent.strip() == "":
            continue
        token_list = []
        span_list = []
        for token in tokenizers[lang](sent):
            if token.text.strip() != "":
                token_list.append(token.text)
                span_list.append((token.idx, token.idx + len(token)))
        tokens.append(token_list)
        spans.append(span_list)
    return tokens, spans


def visualized_prediction(inputs: List[str], prediction: Span, prefix='', lang='pt'):
    spans = list()
    span2event = [[] for _ in inputs]
    for event_idx, event in enumerate(prediction):
        for arg_idx, arg in enumerate(event):
            for token_idx in range(arg.start_idx, arg.end_idx+1):
                span2event[token_idx].append((event_idx, arg_idx))

    for token_idx, token in enumerate(inputs):
        class_labels = ' '.join(
            ['token'] + [f'{prefix}-arg-{event_idx}-{arg_idx}' for event_idx, arg_idx in span2event[token_idx]]
        )
        spans.append(f'<span id="{prefix}-token-{token_idx}" class="{class_labels}" style="background-color">{token} </span>\n')

    for event_idx, event in enumerate(prediction):
        spans[event.start_idx] = (
            f'<span class="highlight bottom blue" '
            f' onmouseenter="highlight_args({event_idx}, \'{prefix}\')" onmouseleave="cancel_highlight(\'{prefix}\')">'
            '<span class="highlight__content" align="center">'
            f'<span class="event" id="{prefix}-event-{event_idx}">'
            + spans[event.start_idx]
        )
        spans[event.end_idx] += f'</span></span><span class="highlight__label"><center>{names[lang][event.label]}</center></span>'
        arg_tips = []
        for arg_idx, arg in enumerate(event):
            arg_tips.append(f'<span class="{prefix}-arg-{event_idx}-{arg_idx}">{names[lang][arg.label]}</span>')
        if len(arg_tips) > 0:
            arg_tips = '<br>'.join(arg_tips)
            spans[event.end_idx] += f'<span class="highlight__tooltip">{arg_tips}</span>\n'
        spans[event.end_idx] += '\n</span>'
    return(
            '<div class="passage model__content__summary highlight-container highlight-container--bottom-labels">\n' +
            '\n'.join(spans) + '\n</div>'
    )


def structured_prediction(inputs, prediction, lang='pt'):
    ret = list()
    for event in prediction:
        event_text, event_label = ' '.join(inputs[event.start_idx: event.end_idx+1]), names[lang][event.label]
        ret.append(f'<li class="list-group-item list-group-item-info">'
                   f'<strong>{event_label}</strong>: {event_text}</li>')
        for arg in event:
            arg_text = ' '.join(inputs[arg.start_idx: arg.end_idx+1])
            ret.append(
                f'<li class="list-group-item">&nbsp;&nbsp;&nbsp;&nbsp;<strong>{names[lang][arg.label]}</strong>: {arg_text}</li>'
            )
    content = '\n'.join(ret)
    return '\n<ul class="list-group">\n' + content + '\n</ul>'


def include_char_spans(annotations, char_spans):
    for annotation in annotations:
        start_char = char_spans[annotation["span"][0]][0]
        end_char = char_spans[annotation["span"][1]][1]
        annotation["char_span"] = [start_char, end_char]
        if "children" in annotation:
            include_char_spans(annotation["children"], char_spans)


@app.route('/')
def sftp():
    ret = template
    input = request.args.get('sentence')
    lang = request.args.get('lang')
    ret = ret.replace('TIMESTAMP', ds_meta["timestamp"])
    if input is not None:
        ret = ret.replace('DEFAULT_SENTENCE', input)
        tokens, _ = tokenize(input.split('\n'))
        model_outputs = predictor.predict_batch_sentences(tokens, max_tokens=512)
        # model_outputs[0].span.tree(model_outputs[0].sentence)
        vis_pred, str_pred = list(), list()
        for sent_idx, output in enumerate(model_outputs):
            vis_pred.append(visualized_prediction(output.sentence, output.span, f'sent{sent_idx}', lang))
            str_pred.append(structured_prediction(output.sentence, output.span, lang))
        ret = ret.replace('VISUALIZED_PREDICTION', '<hr>'.join(vis_pred))
        ret = ret.replace('STRUCTURED_PREDICTION', '<hr>'.join(str_pred))
    else:
        ret = ret.replace('DEFAULT_SENTENCE', default_sentence)
        ret = ret.replace('VISUALIZED_PREDICTION', '')
        ret = ret.replace('STRUCTURED_PREDICTION', '')
    return ret


@app.route('/parse', methods=['POST'])
def parse():
    try:
        data = request.get_json(force=True)
        sentences = data.get('sentences', [])
        lang = data.get('lang', 'en')

        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            return jsonify({'error': 'Invalid input: "sentences" must be a list of strings.'}), 400
        if lang not in tokenizers:
            return jsonify({'error': f'Invalid language: "{lang}". Must be one of {list(tokenizers.keys())}.'}), 400

        tokens, char_spans = tokenize(sentences, lang=lang)
        model_outputs = predictor.predict_batch_sentences(tokens, max_tokens=512)
        response_data = []
        for i, output in enumerate(model_outputs):
            sent_tokens = output.sentence
            annotations = output.span.to_json()["children"]
            include_char_spans(annotations, char_spans[i])
            response_data.append({
                'tokens': sent_tokens,
                'annotations': annotations
            })

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


app.run(host='0.0.0.0', port=args.p)
