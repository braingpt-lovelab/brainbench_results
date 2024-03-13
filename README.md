# brainbench_results
Raw results and plotting scripts for paper https://arxiv.org/abs/2403.03230

### To work with this repo locally:
```
git clone git@github.com:braingpt-lovelab/brainbench_results.git --recursive
```

### To plot figures in the paper:
* Fig. 3A: `python overall_accuracy_model_vs_human.py`
* Fig. 3B: `python accuracy_by_subfields.py`
* Fig. 3C: `python accuracy_by_positions.py`
* Fig. 5: `python finetuning_boost.py`
* (Will support other figures in near future).

### To obtain human results:
Please refer to the dedicated repo - [https://github.com/braingpt-lovelab/brainbench_participant_data/tree/main](https://github.com/braingpt-lovelab/brainbench_participant_data/tree/a819a1b3766abe4817b1ef81ebe7a0a7a351aa99)

### To obtain model results:
Model perplexities on BrainBench testcases:
* For human created testcases, see `model_results/<model_name>/human_abstracts/PPL_A_and_B.npy`, which is a 2D numpy array, with shape `(num_testcases, 2)`, `PPL_A_and_B[i][0]` is the perplexity of the first abstract of the ith testcase.
* For GPT-4 created testcases, see `model_results/<model_name>/llm_abstracts/PPL_A_and_B.npy`
* For each testcase's ground truth, see `model_results/<model_name>/<human|llm_abstracts>/labels.npy`, which is a 1D numpy array, with shape `(num_testcases, )`. Specifically, `label=0` means the first abstract of this testcase is the correct answer, and `label=1` means the second abstract of the testcase is the correct answer.

### Attribution
```
@misc{luo2024large,
      title={Large language models surpass human experts in predicting neuroscience results}, 
      author={Xiaoliang Luo and Akilles Rechardt and Guangzhi Sun and Kevin K. Nejad and Felipe Yáñez and Bati Yilmaz and Kangjoo Lee and Alexandra O. Cohen and Valentina Borghesani and Anton Pashkov and Daniele Marinazzo and Jonathan Nicholas and Alessandro Salatiello and Ilia Sucholutsky and Pasquale Minervini and Sepehr Razavi and Roberta Rocca and Elkhan Yusifov and Tereza Okalova and Nianlong Gu and Martin Ferianc and Mikail Khona and Kaustubh R. Patil and Pui-Shee Lee and Rui Mata and Nicholas E. Myers and Jennifer K Bizley and Sebastian Musslick and Isil Poyraz Bilgin and Guiomar Niso and Justin M. Ales and Michael Gaebler and N Apurva Ratan Murty and Chloe M. Hall and Jessica Dafflon and Sherry Dongqi Bao and Bradley C. Love},
      year={2024},
      eprint={2403.03230},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
}
```
