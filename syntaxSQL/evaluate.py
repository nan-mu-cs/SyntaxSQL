from __future__ import division
import os
import argparse
import torch
import codecs
import glob

import table
import table.IO
import opts
from conala_eval import tokenize_for_bleu_eval
import bleu_score

UNK_WORD = '<unk>'
RIG_WORD = '<]>'
LFT_WORD = '<[>'

parser = argparse.ArgumentParser(description='evaluate.py')
opts.translate_opts(parser)
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)
opt.anno = os.path.join(opt.root_dir, opt.dataset, '{}.json'.format(opt.split))
opt.pre_word_vecs = os.path.join(opt.root_dir, opt.dataset, 'embedding')

if opt.beam_size > 0:
    opt.batch_size = 1


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    js_list = table.IO.read_anno_json(opt.anno, opt)

    metric_name_list = ['tgt']
    prev_best = (None, None)
    for fn_model in glob.glob(opt.model_path):
        opt.model = fn_model
        print(fn_model)
        print(opt.anno)

        translator = table.Translator(opt, dummy_opt.__dict__)
        data = table.IO.TableDataset(
            js_list, translator.fields, 0, None, False)
        test_data = table.IO.OrderedIterator(
            dataset=data, device=opt.gpu, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)

        # inference
        r_list = []
        for batch in test_data:
            r = translator.translate(batch)
            r_list += r
        r_list.sort(key=lambda x: x.idx)
        assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(
            len(r_list), len(js_list))

        # evaluation
        ref = []
        hyp = []
        print("\n------------ref and hyp---------------\n")
        for pred, gold in zip(r_list, js_list):
            str_map_nl = gold["str_map_nl"]
            g_code = []
            for g in gold["tgt"]:
                if g != RIG_WORD and g != LFT_WORD:
                    if "STR" in g:
                        for k, v in str_map_nl.items():
                            if v == g:
                                g_code.append(k)
                                break
                    else:
                        g_code.append(g)
            p_code = []
            for g in pred.tgt:
                if g != RIG_WORD and g != LFT_WORD:
                    if "STR" in g:
                        for k, v in str_map_nl.items():
                            if v == g:
                                p_code.append(k)
                                break
                    elif g == UNK_WORD:
                        p_code.append("UNK")
                    else:
                        p_code.append(g)

            ref.append(tokenize_for_bleu_eval(" ".join(g_code)))
            hyp.append(tokenize_for_bleu_eval(" ".join(p_code)))
            print("--------------------------------")
            print("ref: ", " ".join(g_code))
            print("hyp: ", " ".join(p_code))
                       
        bleu_tup = bleu_score.compute_bleu([[x] for x in ref], hyp, smooth=False)   
        bleu = bleu_tup[0]    
        exact = sum([1 if h == r else 0 for h, r in zip(hyp, ref)])/len(hyp)
        #pred.eval(gold)
        print('Results:')
        for metric_name in metric_name_list:
            #c_correct = sum((x.correct[metric_name] for x in r_list))
            #acc = c_correct / len(r_list)
            print('{}: {}, {:.2%}'.format(metric_name, bleu, exact))
            if metric_name == 'tgt' and (prev_best[0] is None or bleu > prev_best[1]):
                prev_best = (fn_model, bleu)

    if (opt.split == 'dev') and (prev_best[0] is not None):
        with codecs.open(os.path.join(opt.root_dir, opt.dataset, 'dev_best.txt'), 'w', encoding='utf-8') as f_out:
            f_out.write('{}\n'.format(prev_best[0]))


if __name__ == "__main__":
    main()
