# MM_VQA_RelationNet

**Relation Network on Sort-of-CLEVR - convergence study (PAI-SG Multimodal Wave 2)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One model of the visual question answering convergence comparison in [MM_Wave2_SortOfCLEVR](https://github.com/Statera-Workshop/MM_Wave2_SortOfCLEVR).

## Method

Relation Network (Santoro et al., NeurIPS 2017): the same CNN; each of the 25 cells (+ x, y) is an object; g(o_i, o_j, q) = MLP 4 x 256 over all 625 pairs, summed; f = MLP 256-256 (dropout 0.5)-10.

**Common setup** (identical for the five Wave 2 repositories; full comparison in [MM_Wave2_SortOfCLEVR](https://github.com/Statera-Workshop/MM_Wave2_SortOfCLEVR)).
Sort-of-CLEVR generated with `code/gen_sortofclevr.py` (no download): 75 x 75 RGB images with 6 objects (one per
colour, square or circle), 10 questions per image as 11-d vectors (colour, relational or not, subtype), 10 answers;
5 non-relational questions (shape, left?, top?) and 5 relational ones (shape of the closest / furthest object, count of
the same shape). Train 10,000 images (100,000 questions), test 2,000 (20,000). All models from scratch: Adam 1e-4
constant, batch 64, budgets 20 and 60 epochs, seeds 777 / 778, fp32, deterministic cuDNN. Test accuracy is evaluated
twice per epoch below 30 epochs and every epoch otherwise, each time in plain eval mode and after re-estimating the
BatchNorm statistics on 50 training batches (the series' evaluation practice); the two agree within 0.004 here.
Reported value = last evaluation.

## Results (Sort-of-CLEVR test, mean of seeds; range in brackets)

| Model | Epochs | Seeds | Final accuracy, plain (range) | Final, BN re-estimated (range) | Relational | Non-relational | Peak (epoch) | E90 (epochs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Relation Network | 20 | 2 | 0.689 (0.032) | 0.690 (0.034) | 0.728 | 0.651 | 0.691 (19.8) | 7.0 |
| Relation Network | 60 | 2 | 0.914 (0.012) | 0.914 (0.013) | 0.842 | 0.986 | 0.915 (58.0) | 34.0 |

**Finding.** Slow start (about the baseline level at 20 epochs) but the best model at 60 epochs and the best on relational questions; still rising at the end of the 60-epoch budget.

## Weights

Release **v1.0** (4 files, SHA-256 in `SHA256SUMS.txt`): `sortofclevr_rn_e20_s777.pth`, `sortofclevr_rn_e20_s778.pth`, `sortofclevr_rn_e60_s777.pth`, `sortofclevr_rn_e60_s778.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_c as M
m = M.build('rn')
m.load_state_dict(torch.load('sortofclevr_rn_e60_s777.pth', weights_only=True), strict=True)
```

## Run

```bash
cd code && bash smoke_c.sh        # generates the data, gates, 1-epoch smoke
python mm_c.py --model rn --epochs 60 --seed 777
python analyze_c.py
```
