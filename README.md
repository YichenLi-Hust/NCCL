# NCCL Benchmark

A PyTorch benchmark for federated continual learning. Run methods through one command and compare them under a specified task stream, client schedule and evaluation rule. The synchronized image benchmark follows Section VII of *Unleashing the Power of Continual Learning on Non-Centralized Devices*. The collection also covers asynchronous task arrival, concept drift, few-shot class increments, decentralized learning and continual test-time adaptation. [Survey](https://doi.org/10.1109/COMST.2025.3568279)

The collection has **28 native methods**, **15 author-checkout launchers** and **1 paper-only entry**. The CLI also provides `fedavg-static` as a fixed-data control.

## Quick start

Run from the repository root using Python 3.10–3.12. The offline examples were tested on CPU with Python 3.12 and PyTorch 2.2.2.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m nccl_benchmark list
python -m nccl_benchmark run fedavg --profile smoke --gpu -1
python -m nccl_benchmark run fedrnc --profile smoke --gpu -1
python -m nccl_benchmark catalog fedrnc
python -m nccl_benchmark summarize results > scores.csv
```

Use `run <id> --help` for method-specific arguments. `list` shows executable entries and their protocols. `catalog [id]` shows paper details and implementation status. The native `smoke` profile uses synthetic data without downloads. `default` uses method defaults. `reference` loads a configuration for the chosen method and dataset. For synchronized methods without a dedicated preset, it uses the Survey settings below. CLI flags override preset values.

## Methods

**Native** runs a benchmark implementation in this repository. **Author checkout** launches the linked implementation in a separately prepared environment. **Paper** records a published or accepted method awaiting an executable implementation. A method's task definition and inputs determine which comparisons are meaningful.

| Method and CLI ID | Publication | Benchmark setting or mechanism | Execution |
| --- | --- | --- | --- |
| FedAvg + replay · `fedavg` | [AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html) | Sample-weighted FedAvg with benchmark-local exemplar replay | Native |
| FedWeIT · `fedweit` | [ICML 2021](https://proceedings.mlr.press/v139/yoon21b.html) | Sparse task-specific parameters and attentive transfer | Native |
| GLFC · `glfc` | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Dong_Federated_Class-Incremental_Learning_CVPR_2022_paper.html) | Class-balanced replay, gradient compensation and proxy-based teacher selection | Native |
| CFeD · `cfed` | [IJCAI 2022](https://www.ijcai.org/proceedings/2022/303) | Learning and preservation clients with public surrogate data | Native |
| TARGET · `target` | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_TARGET_Federated_Class-Continual_Learning_via_Exemplar-Free_Distillation_ICCV_2023_paper.html) | Server-side data synthesis and client distillation | Native |
| FedCIL · `fedcil` | [ICLR 2023](https://openreview.net/forum?id=cRxYWKiTan) | Conditional generative replay and consistency loss | Native |
| MFCL · `mfcl` | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d160ea01902c33e30660851dfbac5980-Abstract-Conference.html) | Server generator and classifier calibration | Native |
| LANDER · `lander` | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tran_Text-Enhanced_Data-free_Approach_for_Federated_Class-Incremental_Learning_CVPR_2024_paper.html) | Language-guided data-free rehearsal | [Author checkout](https://github.com/tmtuan1307/LANDER) |
| Re-Fed · `re-fed` | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Towards_Efficient_Replay_in_Federated_Incremental_Learning_CVPR_2024_paper.html) | Personalized gradient-based replay selection | Native |
| TagFed · `tagfed` | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Traceable_Federated_Continual_Learning_CVPR_2024_paper.html) | Trace and recover knowledge of recurring tasks | [Author checkout](https://github.com/P0werWeirdo/TagFCL) |
| pFedDIL · `pfeddil` | [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6262_ECCV_2024_paper.php) | Personalized domain matching and ensemble inference | Native |
| AF-FCL · `af-fcl` | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd1810bba6fd3d8b7a6f4aa7c263c591-Abstract-Conference.html) | Conditional feature-flow replay with relevance weighting | Native |
| FOT · `fot` | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf53f12ab4ea81d942f16162f7f37e6c-Abstract-Conference.html) | Client activation sketches and server update projection | Native |
| Powder · `powder` | [ICML 2024](https://proceedings.mlr.press/v235/piao24a.html) | Cross-task and cross-client prompt transfer | [Author checkout](https://github.com/piaohongming/Powder) |
| SR-FDIL · `sr-fdil` | [IEEE TPDS 2024](https://doi.org/10.1109/TPDS.2024.3436874) | Local and global replay scoring and client sampling | Native |
| CoLA · `cola` | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/de0e668df3fe63ec89e5a7e68f3d350f-Abstract-Conference.html) | Decentralized, lifelong test-time adaptation on shifting device streams | [Author checkout](https://github.com/Cascol-Chen/COLA) |
| FedCCFA · `fedccfa` | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/942e820be4aa112509b3a281ff398851-Abstract-Conference.html) | Client groups and class anchors under distributed concept drift | [Author checkout](https://github.com/Chen-Junbao/FedCCFA) |
| FedTA · `fedta` | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Handling_Spatial-Temporal_Data_Heterogeneity_for_Federated_Continual_Learning_via_Tail_CVPR_2025_paper.html) | Tail-anchor stabilization under client and task heterogeneity | [Author checkout](https://github.com/SkyOfBeginning/FedTA_CVPR2025) |
| pFedMxF · `pfedmxf` | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_pFedMxF_Personalized_Federated_Class-Incremental_Learning_with_Mixture_of_Frequency_Aggregation_CVPR_2025_paper.html) | Personalized class increments with frequency-domain parameter aggregation | Native |
| Hybrid Rehearsal (HR) · `hr` | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d2511dfb731fa336739782ba825cd98c-Abstract-Conference.html) | Latent exemplars, class anchors and synthetic rehearsal | Native |
| LoRM · `lorm` | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d79b8c42f61b51722992705022134ef3-Abstract-Conference.html) | Closed-form LoRA merging for vision Class-IL | [Author checkout](https://github.com/aimagelab/fed-mammoth) |
| Online-FCL · `online-fcl` | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/23c9c94227f937cfb50592a15e7fbb63-Abstract-Conference.html) | Online streams and uncertainty-aware memory | [Author checkout](https://github.com/MLO-lab/online-FCL) |
| UOPP · `uopp` | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f6e3da09789adbebdace028379515568-Abstract-Conference.html) | Few-shot novel classes with prompts and ODE-updated prototypes | Native |
| CAN · `can` | [ICML 2025](https://proceedings.mlr.press/v267/rong25a.html) | Class-specific client expertise guides generative rehearsal | Native |
| FedSSI · `fedssi` | [ICML 2025](https://proceedings.mlr.press/v267/li25cq.html) | Rehearsal-free Class-IL and Domain-IL | Native |
| C²Prompt · `c2prompt` | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e8f5d2924f1bff724588eb0f8f5e62d-Abstract-Conference.html) | Class-aware prompt sharing for pretrained vision models | [Author checkout](https://github.com/zhoujiahuan1991/NeurIPS2025-C2Prompt) |
| DCFCL · `dcfcl` | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a2582a9f63edd4dd9489bd59ed0b6fce-Abstract-Conference.html) | Decentralized personalized FCL | [Author checkout](https://github.com/ydn3229/DCFCL) |
| FedCBDR · `fedcbdr` | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d611d06e3207330555fbc10810e70163-Abstract-Conference.html) | Class-balanced constrained replay in Class-IL | Native |
| MOTION · `motion` | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/659afd140447b7d50fadf8bde6a2240d-Abstract-Conference.html) | Continual node classification on federated graphs with evolving graph coarsening | [Author checkout](https://github.com/GuanchengWan/MOTION) |
| MultiFCL · `multifcl` | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/20de741d21f1a038093c6e3ee7c09481-Abstract-Conference.html) | FCL of pretrained backbones with multi-scale experts | [Author checkout](https://github.com/yang12318/MultiFCL) |
| FedRNC · `fedrnc` | [AAAI 2026](https://doi.org/10.1609/aaai.v40i26.39359) | Class increments with client-specific label noise and progressive correction | Native |
| BPFedCTTA · `bpfedctta` | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Towards_Stable_Federated_Continual_Test-Time_Adaptation_in_Wild_World_CVPR_2026_paper.html) | Sequential unlabeled target clients with entropy-gated updates | Native |
| FEAT-F · `feat` | [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Qi_From_Selection_to_Scheduling_Federated_Geometry-Aware_Correction_Makes_Exemplar_Replay_CVPR_2026_paper.html) | Exemplar replay with simplex geometry and global tail calibration | Native |
| Fed-Duet · `fed-duet` | [ICLR 2026](https://openreview.net/pdf?id=Jk8g1OxyZY) | Continual federated vision-language adaptation | [Author checkout](https://github.com/cocogt96/Fed-Duet) |
| FEROMA · `feroma` | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7565f036ceb20a2c74d341bfaa9fffad-Abstract-Conference.html) | Client profile mapping under temporal distribution drift | [Author checkout](https://github.com/dariofenoglio98/FEROMA) |
| CLIF-R · `clif` | [ICML 2026](https://proceedings.mlr.press/v306/qi26n.html) | Cross-view Lewis-weight exemplar replay in Class-IL | Native · [CLIF-F source](https://github.com/qizhuang-qz/CLIF) |
| C²-AFCL · `c2afcl` | [ICML 2026](https://proceedings.mlr.press/v306/li26gx.html) | Asynchronous task arrival with two-stage update calibration | Native |
| DCU · `dcu` | [ICML 2026](https://proceedings.mlr.press/v306/zhang26l.html) | Peer-to-peer class increments with explicit class unlearning | Native |
| GFedCL · `gfedcl` | [ICML 2026](https://proceedings.mlr.press/v306/yu26q.html) | Spatio-temporal graph-based replay | [Author checkout](https://github.com/IntelliSys-Lab/GFedCL) |
| Li-FIL · `li-fil` | [ICML 2026](https://proceedings.mlr.press/v306/wang26du.html) | Decoupled feature replay in Class-IL | Native |
| RC-FCL · `rc-fcl` | [ICML 2026](https://proceedings.mlr.press/v306/su26l.html) | Asynchronous concept drift with class-conditional references and Sinkhorn detection | Native |
| SPECIAL · `special` | [ICML 2026](https://proceedings.mlr.press/v306/xu26af.html) | Domain-IL with partial participation | Native |
| Ha-FCL · `ha-fcl` | [NeurIPS 2026 accepted](https://openreview.net/forum?id=6F5SkPuDh3) | Width-aware training with spatial and temporal distillation | Paper |
| SPR-FIL · `spr-fil` | [NeurIPS 2026 accepted](https://openreview.net/forum?id=v2AOJ4JiUk) | Rehearsal-free statistical prototypes and constrained encoder updates | Native |

NeurIPS 2026 entries are marked accepted where the conference has announced its decisions. [Conference program](https://neurips.cc/Downloads/2026)

### Native experiments

For a matched comparison under label noise, hold the seed, task order, client partition and noise rate fixed. FedRNC defaults to 20% target noise. Other synchronized methods default to clean labels. The noise generator modifies training shards and records realized corruption counts per client. Test labels remain clean.

```bash
for method in fedavg fedrnc; do
  python -m nccl_benchmark run "$method" --profile reference --dataset cifar100 \
    --data-dir ./data --download --seed 7 --class_noise_rate 0.2 --gpu 0
done
```

FedRNC performs two extra client warm-up phases per task. For matched compute, account for these additional local updates. For a common training budget, also match the backbone, client participation, epochs, rounds and memory.

```bash
python -m nccl_benchmark run uopp --profile reference --dataset cifar100 \
  --data-dir ./data --download --uopp_shots 5 --gpu 0
python -m nccl_benchmark run dcu --profile smoke --dcu_forget_after 2:0,3:4 --gpu -1
python -m nccl_benchmark run bpfedctta --dataset cifar10c --data-dir ./data --gpu 0
```

### Author checkouts

Follow the linked repository instructions for its Python environment, datasets and pretrained weights. Pass the checkout and its Python executable to NCCL. Use `--dry-run` to inspect the resolved directory and exact arguments. Python entry points accept author-specific arguments after `--`.

```bash
git clone https://github.com/tmtuan1307/LANDER ../LANDER
python -m nccl_benchmark run lander --source-dir ../LANDER \
  --python /path/to/lander-env/bin/python --dry-run
python -m nccl_benchmark run lander --source-dir ../LANDER \
  --python /path/to/lander-env/bin/python --output-dir ./results/lander-run
```

| Group | Default experiment and preparation |
| --- | --- |
| LoRM, GFedCL, Fed-Duet | LoRM uses the rank-one CIFAR-100 configuration with a pretrained ViT. GFedCL launches its CIFAR-100 entry point. Fed-Duet runs the CIFAR-100 10-increment Hydra configuration from `cil/`. |
| TagFed, LANDER, FedTA, Powder | TagFed uses its legacy environment in `src/`. LANDER uses text-enhanced synthesis on five CIFAR-100 tasks. FedTA needs prepared CIFAR-100 and ViT weights. Powder starts its ImageNet-R experiment from `src/`. |
| Online-FCL, MultiFCL, DCFCL, C²Prompt | Online-FCL uses entropy-based memory management. MultiFCL and DCFCL use their released training drivers and dataset layouts. C²Prompt runs its upstream fixed ImageNet-R shell script. |
| MOTION, FedCCFA | MOTION runs continual node classification on Cora and requires the repository's graph-learning environment. FedCCFA runs its distributed concept-drift experiment from `methods/`. |
| FEROMA, CoLA | FEROMA needs a configured Flower experiment and the data for the selected fold. Use `-- --prepare-data --fold 0` on a prepared checkout to generate a fold. CoLA uses the collaborative ETA configuration, ImageNet target domains and pretrained ViT weights. |

Each launcher records its command, source path, entry-point hash, available Git revision, exit status and log. The author configuration controls the destination and units of model outputs and metrics.

## Data and reference settings

Use `--data-dir` for local data, `--download` for supported torchvision datasets, `--gpu -1` for CPU or `--gpu 0` for a CUDA device. For asynchronous vision methods, `--gpus 0,1` distributes client work over two CUDA devices.

| Native dataset | Default synchronized task stream | Data layout |
| --- | --- | --- |
| `cifar10`, `cifar100` | 5 × 2 classes, 10 × 10 classes | Standard torchvision CIFAR folders |
| `tinyimagenet` | 10 × 20 classes | `train/<wnid>/images/`, `val/images/`, `val/val_annotations.txt` |
| `digit10` | MNIST → EMNIST Digits → USPS → SVHN | Torchvision datasets under their respective lowercase subdirectories |
| `office31` | Amazon → Webcam → DSLR | `<domain>/[images/]<class>/<image>` |
| `officecaltech10` | Amazon → Caltech → Webcam → DSLR | Same class folder names across four domains |
| `cifar10c`, `cifar100c` | Clean source → 15 corruption domains for BPFedCTTA | Torchvision clean CIFAR plus `CIFAR-10-C/` or `CIFAR-100-C/` containing `labels.npy` and the official corruption arrays |

Tiny-ImageNet uses its labeled validation split. Office datasets use a seeded per-class train/test split and 224-pixel crops. Digit10 uses 32×32 RGB images to preserve SVHN color. The Survey reports 28×28 grayscale inputs. For UOPP, CIFAR-100 uses 60 base classes and eight five-class increments. Its shot budget caps each client's access to each novel class for the whole task.

| Survey reference configuration | Clients / active | Rounds per task | Local epochs | Batch | Learning rate | Dirichlet α | Top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 20 / 8 | 80 | 20 | 64 | .01 | .1 | 1 |
| CIFAR-100 | 20 / 10 | 100 | 20 | 64 | .01 | 1 | 1 |
| Tiny-ImageNet | 20 / 12 | 100 | 20 | 128 | .001 | 10 | 10 |
| Digit10 | 15 / 6 | 60 | 20 | 64 | .001 | .1 | 1 |
| Office-31 | 10 / 4 | 60 | 20 | 32 | .01 | 1 | 1 |
| Office-Caltech-10 | 8 / 4 | 40 | 15 | 32 | .01 | 1 | 1 |

Survey runs use ResNet-18 and a 300-example replay budget for methods that store examples. Dedicated method presets and model requirements are recorded in the effective run configuration. The Survey reports two repeats. Run each seed separately and retain the individual manifests.

## Results and metrics

Native runs write `run.json`, `metrics.csv` and `model.pt` under `results/<method>/<dataset>/seed<seed>-<timestamp>/`. Pass `--output-dir` to select an empty directory. The manifest records the method, profile, configuration, package versions, protocol and filenames. Use `summarize` on a single run or a results tree to export a CSV with the metric name, unit, method, dataset, protocol and seed. Author checkouts contribute their launch status.

| Protocol | What to read |
| --- | --- |
| Synchronized image tasks | `metrics.csv` gives sample-weighted top-k accuracy on all seen tasks. `task_metrics.csv` gives each task's accuracy. In `summary.json`, `task_end_accuracy_matrix[i][j]` is accuracy on task *j* after stage *i*. Unseen entries are null. `final_task_macro_accuracy` averages the final row. `final_average_forgetting` averages each older task's best **earlier stage-end** accuracy minus its final accuracy. A gain can yield negative forgetting. One task yields null. These metrics are fractions. |
| Survey reporting convention | `final_accuracy` averages up to the last ten evaluations of the final stage. `average_accuracy` averages that statistic across stages. `rounds_to_best_sum` sums the first best round of each stage. |
| Asynchronous task arrival | `AC` is the macro average over released tasks, in percent. `FS` measures accuracy drops from previous bests on older tasks, in percentage points. `events.jsonl` and `data_coverage.json` record arrivals, accepted updates and coverage. |
| Asynchronous concept drift | `AC` is pooled accuracy in percent. `FS` measures forgetting across seen tasks. `FS_old` restricts it to earlier tasks. `protocol.json` records affected clients, drift transforms and delays. |
| Continual test-time adaptation | BPFedCTTA reports source, current-domain and pooled seen-domain accuracy as fractions. `arrivals.csv` records each incoming client's uncertainty and aggregation weight. |
| Decentralized unlearning | DCU records retained-class and requested-class accuracy for training and forgetting events. `model.pt` stores peer models and the set of removed classes. |


## Implementation choices

- **Task timing.** C²-AFCL has reference presets for CIFAR-100, Tiny-ImageNet, Office-31 and Office-Caltech-10. Its Office-Caltech-10 order is Amazon → Webcam → DSLR → Caltech. RC-FCL supports CIFAR-10, CIFAR-100, Tiny-ImageNet and a Digit10 mixture supplied as ten matching class folders under `train/` and `test/`. Use `--rc_iid`, `--alpha 0.5` or `--alpha 0.1` for its partition variants.
- **Public data and proxies.** CFeD requires independent surrogate images for real multi-task runs via `--cfed_surrogate_dir`. GLFC ranks historical teachers using class feature centroids on the synchronized stream. Its paper uses a gradient-based proxy and independent client task changes.
- **Replay and parameter isolation.** AF-FCL uses a native PyTorch conditional flow. TARGET, FedCIL, MFCL and CAN use bounded synthesis budgets. FOT exposes activation and sketch caps. FedWeIT stores sparse task adapters and simulates masked communication with dense aggregation buffers. pFedDIL evaluates aggregated personalized ensembles. SR-FDIL uses 32×32 adversarial examples and CNN feature prototypes.
- **Client and feature statistics.** FedSSI consolidates each client's shard at task boundaries. Li-FIL uses capped, noisy feature uploads with a conditional VAE. FedCBDR pools feature Gram matrices in the simulator. CLIF implements the CLIF-R replay variant. FEAT implements FEAT-F with the FedCBDR exemplar selector. The Li-FIL and FedCBDR ports simulate statistics exchange without measuring end-to-end privacy guarantees.
- **Domain increments.** SPECIAL's Digit10 reference uses USPS → SVHN → EMNIST → MNIST. BPFedCTTA trains on a labeled source federation before each target client adapts to unlabeled images. The classification port uses past and current arrivals for the uncertainty gate. On CIFAR-C, a fixed split of underlying image identities separates adaptation from evaluation across all corruptions. Office and Digit10 are alternative benchmark streams.
- **New-class rehearsal.** HR aggregates class-centroid reports at task arrival and caps total latent codes per client. CAN estimates expertise per class on client shards and adapts the amount of generated replay. FedRNC applies local loss-mixture filtering and prototype correction on synchronized class tasks.
- **Pretrained backbones.** UOPP accepts local torchvision ViT-B/16 weights through `--uopp_weights` or downloads with `--download`. Its port uses Dirichlet client partitions and per-example routing. pFedMxF accepts a backbone-only DINO ViT-B/16 checkpoint through `--pfedmxf_weights` or `--download`. Both use a small frozen transformer in the smoke profile. [DINO checkpoints](https://github.com/facebookresearch/dino)
- **Statistical prototypes.** SPR-FIL implements the homogeneous-backbone variant, sharing class means and covariances, aligning statistics with procedural fractal anchors and projecting encoder gradients. The port defaults to rank 8, 128 anchors and a covariance ridge of 0.0001.
- **Decentralized unlearning.** DCU uses a peer graph, disposable diffusion prototypes and explicit class-removal requests. For real images install `diffusers` and `safetensors`, then supply a local Stable Diffusion 1.x Diffusers directory with `--dcu_diffusion`. The smoke profile uses a small analytic diffusion model. Requests such as `--dcu_forget_after 2:0` remove class 0 after the second task. The CIFAR-100 reference preset takes the first three 10-class blocks and removes classes 0 and 20 after tasks two and three. The latent diffusion checkpoint is a benchmark choice.


## Verification and extension

Run `python -m unittest discover -s tests -v` for the registry, task-matrix and result-export checks. Native integrations have been exercised using offline smoke runs. HR additionally completed ten tasks on a small CIFAR-100 image subset. UOPP's base split and shot limits were checked on the same type of data. Upstream launches have been checked with local execution fixtures. Full-dataset paper accuracies have not been reproduced.

To add a method, enter its publication in `nccl_benchmark/catalog.py`. For a synchronized strategy, implement the hooks in `survey/strategies.py` and register its class once in `registry.py`. A strategy with its own backbone can implement `build_model(args, stream)`. For a different protocol, register a backend accepting an argument list and returning an object with `.args`. For an author checkout, add its entry point and arguments to `upstream.py`. Add the appropriate row above and validate a run through the public CLI.
