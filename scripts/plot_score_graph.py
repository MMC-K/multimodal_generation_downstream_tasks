import sys
import os
import glob
import matplotlib.pyplot as plt

def get_bleu_rouge_score(root_path):
    dirs = glob.glob(root_path+"_*")
    bleu_scores = [0 for _ in dirs]
    rouge_scores = [0 for _ in dirs]
    for p in dirs:
        epoch = int(p.split("_")[-1])
        result_path = os.path.join(p, "generate_result.txt")
        f = open(result_path, mode="r", encoding="utf-8")
        blue_score = -1
        rouge_score = -1
        for l in f.readlines():
            if "'bleu': " in l:
                blue_score = float(l.split("'bleu': ")[1].split(",")[0])
            elif "'rouge1': " in l:
                rouge_score = float(l.split("'rouge1': ")[1].split(",")[0])
        bleu_scores[epoch] = blue_score
        rouge_scores[epoch] = rouge_score
        # print(p, epoch, blue_score, rouge_score)
    return bleu_scores, rouge_scores

all_data = {}

for rp in sys.argv[1:]:
    bleu_scores, rouge_scores = get_bleu_rouge_score(rp)
    all_data[rp]={'bleu':bleu_scores, 'rouge':rouge_scores}

# plot bleu scores:
plt.figure()
for k, v in all_data.items():
    plt.plot(v['bleu'], label=os.path.basename(k))
plt.legend(loc='center left', bbox_to_anchor=(0.5, -0.4))
plt.savefig("figures/bleu_score_output.pdf")


plt.figure()
for k, v in all_data.items():
    plt.plot(v['rouge'], label=os.path.basename(k))
plt.legend(loc='center left', bbox_to_anchor=(0.5, -0.4))
plt.savefig("figures/rouge_score_output.pdf")
