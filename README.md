# next-token-bio

[![tests](https://github.com/aarnavbhat23/next-token-bio/actions/workflows/tests.yml/badge.svg)](https://github.com/aarnavbhat23/next-token-bio/actions/workflows/tests.yml)

When a language model "predicts the next amino acid" or "the next nucleotide", how much of that is biological grammar, and how much is recognising a close relative from the training set?

This repository trains next-token predictors from scratch on **269k Swiss-Prot proteins** and the **E. coli K-12 genome**: a Kneser-Ney n-gram ladder, a 2.0M-parameter LSTM and a 1.9M-parameter GPT, with GPT and LSTM given identical token budgets. Every model is scored on the same positions in two kinds of test protein: **new members of families seen in training**, and **790 whole families never seen in training**. The genome is evaluated on held-out 50 kb blocks, with each base labelled as non-coding or by its codon position.

![Next-token loss for every model](results/figures/fig1_ladder.png)

## What happened

Bits per symbol, lower is better. Uniform guessing scores 4.322 over amino acids and 2.000 over bases.

| model | protein, seen families | protein, held-out families | E. coli DNA, held-out blocks |
|---|---:|---:|---:|
| 1-gram (composition only) | 4.156 | 4.166 | 2.000 |
| best Kneser-Ney n-gram | **3.383** (6-gram) | 4.132 (3-gram) | 1.936 (8-gram) |
| GPT, 1.9M | 4.066 | 4.079 | 1.934 |
| LSTM, 2.0M | 3.995 | **4.059** | **1.882** |

Top-1 accuracy on held-out families is 9.6% for composition alone, 11.7% for the GPT and 12.3% for the LSTM. A seen-family 6-gram reaches 30.6%.

1. **Most apparent protein predictability is memorised homology.** On families seen in training, a 6-gram that simply looks up the previous five residues beats both neural models by more than 0.6 bits. On held-out families the same 6-gram is *worse than uniform guessing* (4.601 bits). The neural models are much less exposed to this: 3.995 vs 4.059 bits for the LSTM.
2. **On unseen families, next-residue prediction is close to the composition floor.** The best model gains 0.107 bits over composition alone, 2.6% of the entropy, which fits the classic finding that protein sequence is nearly incompressible. Per residue, the largest gain is for cysteine (6.23 → 5.82 bits for the GPT), consistent with cysteines occurring in pairs and motifs. The start residue is almost always Met, at about 0.3 bits.
3. **About half of the neural models' long-context gain on proteins is the protein's own composition.** See the composition control below.
4. **In DNA, the predictable base is the last one read in each codon.** The LSTM beats the best n-gram most at exactly those positions, which an 8-gram cannot locate without knowing the reading frame.

![Loss as a function of context length](results/figures/fig2_context.png)

### How much context does each model use?

Each model predicts the same 20,000 held-out targets from only the last *k* symbols; every target has at least 256 symbols of true context. The n-gram is best at 2 residues of protein context and 7 bases of DNA context, then overfits. Both neural models keep improving out to 256.

**Composition control.** A long context could help simply by revealing what kind of protein this is: membrane proteins are hydrophobic throughout, disordered ones rich in S, P, E and K. To test this, the 3-gram (8-gram for DNA) is linearly mixed with a smoothed count of residues in the same *k*-symbol window. The mixing weight and smoothing are tuned per *k* on the **validation** families, then scored on the test targets (dashed line).

| at k = 256 | n-gram alone | + own-composition cache | GPT | LSTM |
|---|---:|---:|---:|---:|
| protein, held-out families | 4.147 | 4.115 | 4.100 | 4.084 |
| E. coli DNA | 1.941 | 1.938 | 1.936 | 1.887 |

For held-out proteins, composition recovers 0.031 of the LSTM's 0.063-bit advantage over the 3-gram. The other ~0.03 bits is something per-protein composition does not explain. These numbers are on positions 256 and beyond, which is why they differ from the full-sequence table. For DNA, composition explains almost none of the LSTM's gain.

![Loss by codon position](results/figures/fig3_codon_phase.png)

### Codon position in E. coli

| bits/nt | non-coding | + codon 1 | + codon 2 | + codon 3 | − codon 1 | − codon 2 | − codon 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| best n-gram (8-gram) | 1.980 | 1.962 | 1.962 | 1.860 | 1.885 | 1.941 | 1.959 |
| LSTM | 1.947 | 1.915 | 1.911 | **1.778** | **1.801** | 1.888 | 1.934 |
| GPT | 1.967 | 1.944 | 1.937 | 1.898 | 1.916 | 1.909 | 1.970 |

The genome is always read 5′→3′ on the + strand. For + strand genes the most predictable base is codon position 3: after two bases of a codon, codon-usage bias narrows the third. For − strand genes the forward reading traverses each codon backwards, so the most predictable base is **codon position 1**, the one read last. The LSTM's advantage over the 8-gram is largest exactly there (0.08 bits, against 0.03 in non-coding DNA). This is the three-base periodicity GeneMark-style gene finders exploit; here a model trained only to predict the next base picks it up, and so it finds part of the reading-frame signal unsupervised.

![Loss along the N-terminus and by residue](results/figures/fig4_position_residue.png)

### Sampling from the GPT

The GPT generated 200 proteins at temperature 1, with mean length 276 against 244 for natural held-out sequences. They contain no homopolymer runs of six or more, have training-like amino acid composition (Jensen–Shannon divergence 0.0008), and share 1.06% of their 10-mers with training. That overlap is at the bitset false-positive floor (1.01%), so the samples are not copies. **This is not evidence of foldable or functional proteins:** composition-matched random sequences pass the same three checks (1.01%, JS 0.00004). Structure prediction or experiments would be needed.

## Limits

* **Small, undertrained models.** About 2M parameters and 13–20M training tokens each, against billions for ProtGPT2 or Evo. The GPT is behind the LSTM here, most likely because Transformers need more tokens to overtake recurrent models at this size. Its validation loss was still falling (4.087 → 4.078 bits over the last 400 steps). Do not read the GPT–LSTM ordering as a general architecture result.
* **One seed, one split.** No confidence intervals. Differences under ~0.01 bits should not be interpreted.
* **Family labels are not a homology guarantee.** The k-mer audit suggests little leakage, but remote homologs across UniProt families can remain.
* **One bacterial genome.** E. coli is ~86% coding; results on eukaryotic genomes with introns and repeats would differ.
* The "best n-gram" is chosen on test data, which favours the baseline.

## Next experiment

Scale the same protocol until the neural models' held-out-family loss stops improving, and check whether the residual ~0.03 bits beyond composition grows. Repeat over several family splits and seeds for intervals. A stronger composition control, conditioning on predicted secondary structure or disorder, would show what the remaining signal is.

## Prior work and credit

Next-token prediction on biological sequence is well established. This repository does not claim a first.

**Autoregressive protein models.** ProGen (Madani et al., *Nature Biotechnology* 2023) generated functional lysozymes from a 1.2B-parameter conditional model. [ProtGPT2](https://doi.org/10.1038/s41467-022-32007-7) (Ferruz et al., *Nature Communications* 2022) is a 738M-parameter GPT-2 trained on about 50M UniRef50 sequences. [ProGen2](https://arxiv.org/abs/2206.13517) (Nijkamp et al., *Cell Systems* 2023) scales the approach to 6.4B parameters; a [2025 review](https://arxiv.org/abs/2501.10282) notes that even the largest ProGen2 still underfits natural sequence. RITA (Hesslow et al., 2022) and Tranception (Notin et al., ICML 2022) study scaling and fitness prediction. Masked models, including ESM-1b (Rives et al., *PNAS* 2021), ESM-2 (Lin et al., *Science* 2023) and ESM3 (Hayes et al., *Science* 2025), use a fill-in-the-blank objective rather than next-token prediction. Cheng et al. (NeurIPS 2024) fit compute-optimal scaling laws for protein language models.

**Genomic models.** DNABERT (Ji et al., *Bioinformatics* 2021) and the Nucleotide Transformer (Dalla-Torre et al., *Nature Methods* 2024) are masked models. HyenaDNA (Nguyen et al., NeurIPS 2023) predicts the next nucleotide over contexts up to 1M bases. Evo (Nguyen et al., *Science* 2024) is a 7B-parameter next-nucleotide model trained on prokaryotic and phage genomes. [Evo 2](https://arcinstitute.org/manuscripts/Evo2.pdf) (Brixi et al., 2025) extends it to 40B parameters, 9.3 trillion bases and a 1M-base context across all domains of life. CaLM (Outeiral & Deane, *Nature Machine Intelligence* 2024) models codon sequences directly.

**Older ideas this repository leans on.** Nevill-Manning & Witten ("Protein is incompressible", DCC 1999) found that general-purpose compressors barely beat the entropy implied by amino acid composition alone. GeneMark (Borodovsky & McIninch, 1993) and Fickett's TESTCODE (1982) use the three-base periodicity of coding DNA to find genes. Modified Kneser-Ney smoothing follows Chen & Goodman (1998).

**What is new here** is the controlled, laptop-scale comparison: the same targets scored by n-grams and parameter-matched neural models, contrasted across seen and held-out families, with a k-mer audit of the split. It is a measurement study, not a new architecture.

## Method

**Protein data.** UniProtKB/Swiss-Prot reviewed entries of 50 to 512 residues (457,140 rows). Sequences with non-standard residues (1,619) and exact duplicates (77,996) are removed. Sequences without a family annotation (37,006) are dropped entirely, so no unlabelled homolog of a held-out family can reach training. The family key is the first clause of UniProt's "Protein families" field, so "Protein kinase superfamily, Ser/Thr protein kinase family" groups at the superfamily level. Families are hashed 80/10/10 into train, validation and **held-out** sets, and 3% of training-family sequences are set aside as the **seen** test set.

| split | sequences | residues | families |
|---|---:|---:|---:|
| train | 269,436 | 73.4M | 6,530 |
| validation (held-out families) | 33,118 | 9.4M | 794 |
| test, seen families | 8,294 | 2.3M | 2,081 |
| test, held-out families | 29,671 | 7.3M | 790 |

Evaluation uses a fixed random 3,000 sequences from each test split. Validation, used only for checkpoint selection, is itself made of held-out families.

**Split audit.** 52.0% of the 10-mers in seen-family test proteins also occur in training, against 1.2% for held-out families. The bitset used for the check has a measured false-positive rate of 1.0%, so true sharing for held-out families is about 0.2%. No held-out sequence shares more than half its 10-mers with training; 51.8% of seen-family sequences do. Family annotation is not a homology guarantee, but this split leaks very little.

**DNA data.** E. coli K-12 MG1655 (NC_000913.3, 4.64 Mb) with RefSeq CDS annotation; 4,303 CDS are used, and 37 pseudogenes or non-multiples of three are skipped. The genome is cut into 50 kb blocks: every tenth block is test and another tenth is validation. Training windows are reverse-complemented half the time. Each scored test base has at least 256 bases of preceding context, for every model.

**Models.**

| model | parameters | notes |
|---|---:|---|
| Kneser-Ney n-gram | — | interpolated modified KN, three discounts; protein orders 1–6, DNA orders 1–12 |
| LSTM | 2.03M | 2 layers, hidden 384 |
| GPT | 1.88M | 4 layers, width 192, 6 heads, learned positions, tied embeddings |

GPT and LSTM are trained for the same steps on the same batches: 1,200 × 32 × 512 = 19.7M tokens for protein, and 800 × 32 × 512 = 13.1M tokens for DNA. AdamW is used with cosine decay, and the best validation checkpoint is kept. Training ran on an Apple M3 GPU (MPS) shared with other heavy jobs, so the token budget was cut from the original plan; see *Limits*.

**Scoring.** Every model's prediction is renormalised over the 20 amino acids or 4 bases before scoring, so special tokens cannot help or hurt. The headline metric is bits per symbol, the cross-entropy in log base 2: 4.32 for uniform guessing over amino acids, 2.00 over bases. Perplexity and top-1 accuracy are also reported. **"Best n-gram" means the order with the lowest *test* loss**, which flatters the baseline; any neural advantage reported here is therefore conservative.

## Reproduce

Python 3.14, PyTorch 2.13 and NumPy 2.5 were used.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
./scripts/download.sh            # ~80 MB from UniProt and NCBI
python -m nexttok.prepare        # splits + data/processed, writes results/data_summary.json
./scripts/train_all.sh           # four neural models (skippable: checkpoints/ is committed)
./scripts/evaluate_all.sh        # every number and figure in this README
```

`results/` holds every JSON the README quotes, the training histories and `summary.md`. The trained checkpoints are in `checkpoints/`, and the held-out accessions in `data/processed/*_accessions.txt` once `prepare` has run. GPT samples are written to `results/generated_proteins.fasta`.

## Data and licences

Code: MIT. UniProtKB data is [CC BY 4.0](https://www.uniprot.org/help/license) (The UniProt Consortium). The E. coli genome and annotation are NCBI RefSeq records, which NCBI places no restrictions on. Raw and processed data are not committed; `scripts/download.sh` fetches them.

Built with AI assistance. All numbers come from the recorded runs in `results/`. There is no independent review or adoption claim.
