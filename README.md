# Germinal Multi: multi-target antibody design

<p align="center">
  <img src="assets/pipeline.png" alt="Multi-target co-optimisation pipeline"/>
</p>

Germinal Multi extends [Germinal](https://github.com/SantiagoMille/germinal) to
**multi-target design**: co-optimizing a single binder against multiple targets
simultaneously. Each design is evaluated against every state during optimization, rather than 
being optimized against one structure and subsequently tested against the others.

## Setup

### Requirements

**Prerequisites:**
- [PyRosetta](https://www.pyrosetta.org/) (academic license required)
- [ColabDesign/AlphaFold-Multimer parameters](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar) (~4 GB; [environment_setup.md](environment_setup.md) section 4 has the download commands)
- [AlphaFold3 parameters](https://github.com/google-deepmind/alphafold3) (the default folding backend, against which the supplied filtering thresholds were calibrated; [Protenix](https://github.com/bytedance/Protenix) is supported as an alternative backend, but the supplied thresholds have not been calibrated for it.)
- JAX with GPU support

> **Note:** AbLang is not yet supported. The only supported antibody language
> model is IgLM (`ablm_model: "iglm"` in every run config). The pipeline
> rejects any other value at startup.

**System Requirements:**
- **GPU**: NVIDIA GPU with CUDA support. This needs a compute node — it will
  not run on a laptop, an Apple-silicon Mac, or a free Colab tier.
- **Memory**: **Recommended GPU memory: 80GB VRAM.** Every shipped configuration co-optimizes a
  ~242-residue scFv against two antigen targets, which means two AlphaFold
  forward and backward passes per step. The 40GB figure often quoted for
  Germinal is an upstream single-target nanobody number and does not apply.
- **Storage**: 50GB+ for results, **plus** several hundred GB for the AF3
  genetic databases, the colabfold MSA database, the AF3 container image and
  the AlphaFold-Multimer parameters (~4GB). Database storage accounts for most of the total storage requirement.
- **On a shared GPU partition, expect occasional `RESOURCE_EXHAUSTED` and
  retry.** Setting `XLA_PYTHON_CLIENT_PREALLOCATE=false` makes JAX allocate on
  demand rather than reserving the card up front, which shares a node better but
  lets a co-tenant process take the memory first. That is contention, not a
  misconfiguration: resubmit, optionally with `--exclude=<node>`. Such a job
  usually dies during its first trajectory and writes no designs — but the
  output path carries no timestamp, so clear or rename
  `results/<experiment_name>/` before retrying rather than appending to it.

### Installation

**See [environment_setup.md](environment_setup.md).** See environment_setup.md (environment_setup.md) for the complete installation instructions,
cloning with submodules, the conda environment, ColabDesign and Germinal,
PyRosetta, the AlphaFold-Multimer and AbMPNN weights, `validate_install.py`,
the optional Protenix backend and the test suite.

**AlphaFold3 is set up separately — see [docs/af3-setup.md](docs/af3-setup.md).**
AF3 was used for filtering designs selected for experimental validation.

## Usage

### Quick Start

The pipeline uses a single entry point, `run_germinal_multi.py`, with execution controlled by the
`logits_only` and `recycling` configuration keys. Germinal Multi requires at least two antigen
targets. Configuration is managed with [Hydra](https://hydra.cc/): `configs/config.yaml` 
defines high-level parameters plus pointers to the
granular settings.

Those settings live in five groups, selected independently on the command line:

 - **Main run settings**: `configs/run/`
 - **Target settings**: `configs/target/`
 - **Post-hallucination (initial) filters**: `configs/filter/initial/`
 - **Final filters**: `configs/filter/final/`
 - **Folding backend, per stage**: `configs/folding/` — also where the AF3 install paths live

The folding backend is selected independently for each pipeline stage. Both
default to AF3. To use Protenix instead, override on the command line:

```bash
python run_germinal_multi.py "folding@filter_folder=protenix" "folding@recycle_folder=protenix"
```

You can mix backends (e.g. Protenix for recycle, AF3 for the filter stages).
See [environment_setup.md](environment_setup.md) § 6 for Protenix installation.

#### Configuration Structure

```
configs/
├── config.yaml                  # Main configuration yaml; msa_db_dir lives here
├── folding/                     # pick ONE per stage; the other is never read
│   ├── af3.yaml                 # AF3 backend + its install paths (all ???)
│   └── protenix.yaml            # Protenix backend
├── run/
│   ├── multi.yaml                # Germinal, generic 242-residue scFv
│   ├── multi_logits.yaml         # Germinal 1.1, generic 242-residue scFv
│   ├── multi_rescue.yaml         # Germinal, bd55_1205 rescue scaffold
│   └── multi_rescue_logits.yaml  # Germinal 1.1, bd55_1205 rescue scaffold
├── target/
│   ├── egfr.yaml                 # EGFR domain III, WT + mutant
│   ├── ba2_kp3.yaml              # SARS-CoV-2 RBD, BA.2 + KP.3
│   └── xfg.yaml                  # XFG RBD, WT + 3-mutation challenge
└── filter/
    ├── initial/
    │   ├── scfv_multi.yaml       # Post-hallucination filters
    │   └── scfv_multi_rescue.yaml  # ...plus an initial.plddt gate
    └── final/
        └── scfv_multi.yaml       # Final acceptance filters
```

#### Configuration compatibility
Run and target configurations must be **compatible**. Each run
configuration specified the binder geometry (`cdr_lengths`, `fw_lengths`, `vh_len`) and, 
for rescue configurations, `complex_path`. These settings must match the binder in the starting
complex. At startup, `config.validate_binder_length` compares
`sum(cdr_lengths) + sum(fw_lengths)` with the binder-chain length and raises an error if they differ. 
It does **not** validate `vh_len`, which is used separately during IgLM preparation. See
[docs/shipped-configs.md](docs/shipped-configs.md) for the pairings that are
known to validate.

> **Note:** these configs are reasonable starting points for parameter
> exploration, not recipes guaranteed to work out of the box on a new target.

To change a run, create or select configuration files, or override individual parameters on the command line. 

### Basic Usage

> **Set the AF3 paths first.** They ship as `???` in
> `configs/folding/af3.yaml`, so a run on the default backends aborts at
> startup naming the missing keys. See [docs/af3-setup.md](docs/af3-setup.md);
> a Protenix-only run needs none of them.

**Run with defaults:**
```bash
python run_germinal_multi.py
```

**Switch to a different run config (e.g., new_config):**
```bash
python run_germinal_multi.py run=new_config
```

**Use different target:**
```bash
python run_germinal_multi.py target=my_target
```

**Use a different config file with Hydra:**
```bash
python run_germinal_multi.py --config-name new_config
```

**Use different filters:**
```bash
python run_germinal_multi.py filter/initial=new_init_filter filter/final=new_final_filter
```

Most keys can be overridden on the command line; see the
[Hydra docs](https://hydra.cc/docs/advanced/override_grammar/basic/) for the
grammar. Two Germinal-specific notes:

- Keys from `configs/run/` are in the global namespace and take no `run.`
  prefix, e.g. `weights_iptm=1.0`.
- **Filter thresholds are not reachable from the command line.** Their keys
  contain a literal dot (`final.plddt`), which Hydra's override grammar cannot
  address under any quoting. Swap the whole set instead
  (`filter/final=scfv_multi`), or copy a filter config and edit the value.

**Example SLURM job script:** see
[examples/submit_job.sbatch](examples/submit_job.sbatch) — copy it, fill in
the `<placeholders>` for your partition and AF3 paths (see
[docs/af3-setup.md](docs/af3-setup.md)), and submit with `sbatch`.

### Target Configuration

For each new target, create a target configuration YAML file. Germinal Multi requires two or more antigen targets, and with one entry per state in each per-target list:

```yaml
# @package target
target_name: ["state_wt", "state_mut"]   # must be unique: these key the per-state columns
target_pdb_path: ["pdbs/state_wt.pdb", "pdbs/state_mut.pdb"]
target_chain: "A"                        # shared across targets
binder_chain: "B"
target_hotspots: ["A43,A98,A133", "A43,A98,A133"]
```

The number of antigen targets is **derived** from these lists, not declared:
`target_name`, `target_pdb_path` and `target_hotspots` must all have the same
length, and startup fails naming the mismatch if they do not. (There is no
`num_targets` key -- it used to duplicate the list length with nothing checking
the two agreed.)

The hotspot lists are per state, so they can differ — useful when a mutation
shifts the numbering of the epitope you want to hold. `target_chain` and
`binder_chain` are shared across all targets.

**The number of antigen targets is determined by the lengths of the per-target lists.** 
The count is derived from the list lengths, and the default `multi_grad_merge_method: "sum"` 
combines the per-target gradients across however many there are. The one exception is
`"pcgrad"`, which merges exactly two; above that it switches to `"sum"` at
startup with a printed note, and the effective value is recorded in
`final_config.yaml`. The shipped campaigns are all two-state.

Overriding a list-valued field from the command line needs **inner quotes on
each entry** — without them Hydra splits on every comma and you get six
single-residue hotspot sets instead of two:

```bash
python run_germinal_multi.py 'target.target_hotspots=["A74,A84,A170","A74,A84,A169"]'
```

### Filters Configuration

Initial filters are applied after hallucination to select designs for sequence redesign. 
Final filters are applied at the end of the pipeline to determine which designs are accepted. 
The initial filter set is a subset of the final filter set. Each filter key must include its 
stage prefix (initial. or final.); keys without a stage prefix are rejected at startup. 
For example:

```yaml
initial.clashes:
  value: 1
  operator: '<'

initial.binder_near_hotspot:
  value: true
  operator: '=='

initial.percent_interface_cdr:
  value: 0.5
  operator: '>'
```

## Output and troubleshooting

[docs/pipeline/](docs/pipeline/) documents the pipeline one document per stage —
hallucination, early check, initial filter, redesign and recycling, final filter
— plus the folding backends, MSA infrastructure and output layout.

Germinal generates organized output directories:

```
results/<experiment_name>/          # results_dir in configs/config.yaml, then experiment_name
├── final_config.yaml           # Complete run configuration after overrides
├── trajectories/               # Designs rejected before final filtering: hallucination, the initial
│   ├── structures/             #   cofolding filters, a failed AbMPNN redesign, or the recycle
│   └── designs.csv             #   ipSAE gate. See failure_stage to tell them apart.
├── filter_failed/              # Reached the final filters and failed them
│   ├── structures/          
│   └── designs.csv           
├── redesign_candidates/        # Structure store: every prediction the filter or
│   ├── structures/             #   recycle model made. Its designs.csv stays empty
│   └── designs.csv             #   by design -- it is not a record directory
├── accepted/                   # Antibodies that pass all filters
│   ├── structures/          
│   └── designs.csv           
├── all_trajectories.csv        # Main CSV: designs from the three record folders above
└── failure_counts.csv          # Per-metric counts of what rejected each trajectory
```

There is no timestamp in the path: a run writes to `results/<experiment_name>/`,
and re-using an `experiment_name` accumulates into the same directory rather
than starting a new one.

**Key Output Files:**
- `accepted/structures/*.pdb` - Final antibody-antigen structure for passing antibody designs.
- `all_trajectories.csv` - Complete list of designs that passed hallucination, their *in silico* metrics, which stage they reached, and the pdb path to the designed structure.

> **Note:** run Germinal as `conda run -n <env> --no-capture-output python -u
> run_germinal_multi.py ...`, as `examples/submit_job.sbatch` does. Without
> `--no-capture-output`, `conda run` buffers stdout and stderr until the
> process exits: the SLURM `.out` file stays empty while the job runs, and if
> the job is cancelled or hits its time limit the output is never flushed at
> all, so a 12-hour run can end with an empty log and no traceback.
>
> `all_trajectories.csv` and the `inflight/` directory are still the better
> way to watch progress, since they remain available if the job is interrupted.

**Hallucination batching.** The pipeline does not fold each hallucination
passer immediately. It accumulates designs that clear the hallucination
confidence gates (iPTM, iPAE, pLDDT) in an `inflight/` queue until either
`max_hallucination_passing` (default 5) passers are collected or
`max_hallucination_attempts` (default 100) total attempts are exhausted.
Only then does it starts the batched cofolding stage — a single batched cofolding call on all
accumulated designs that pass the hallucination filters. This means the pipeline can appear idle for hours while
hallucination runs; `inflight/` growing is the sign of progress.

Raise `max_hallucination_passing` for a target whose gates pass most designs,
to amortize the fold start-up over more of them. Leave it low, or lower it
further, for a target that rejects most: a batch that never fills folds
nothing, and a run can end having parked every passer it produced without
scoring one. The end-of-run summary names this case when it happens.

## Important Notes and Tips for Design

**Tweaking Parameters:**

Optimal design parameters are different for each target and antibody type! If you are experiencing low success rates, we recommend tweaking interface confidence weights (ipTM / iPAE), structure-based weights (helix, beta, framework loss), or the IgLM weights defined in `iglm_scale`.  In particular we recommend playing around with:

```python
weights_plddt: 1.0
weights_pae_inter: 0.5
weights_iptm: 0.7
weights_helix: 0.1
weights_beta: 0.1
framework_contact_offset: 1
```

`iglm_scale` is a key parameter that controls the influence of IgLM during different stages of the design process. `iglm_scale` is defined as a list of four scalar values: `[v_1,v_2,v_3,v_4]`. During the logits phase, iglm_scale increases linearly between v_1 and v_2. During the softmax phase, iglm_scale takes the value of v_3, and during the semi-greedy stage iglm_scale takes the value of v_4. 

Filters are also easily changeable in the filters configurations. To add or remove filters from the initial and final filtering rounds, simply add an entry named `<stage>.<metric>` -- `initial.` or `final.` -- and specify the threshold value and the operator (<, >, =, etc).

Finally, using omit_AAs (e.g., `omit_AAs: "C,A"`), you can now omit any amino acid from appearing in the CDRs, opposed to all of the protein.

An example of a param sweep could be:

```bash
python run_germinal_multi.py weights_beta=0.1 weights_helix=0.1 weights_plddt=1.0 experiment_name=beta01-helix01-plddt1

python run_germinal_multi.py weights_beta=0.3 weights_helix=0.2 weights_plddt=1.5 experiment_name=beta03-helix02-plddt1.5
...
```

## Shipped Configurations

The configurations for the paper's campaigns — three targets, each for both
design styles — are documented in
[docs/shipped-configs.md](docs/shipped-configs.md): the full command for each
one, which run/target/filter configs it pairs with, and what the rescue configs
do differently.

## Citation

If you use this work, please cite:

```bibtex
@article{park_antibody_2026,
	title = {Antibody design across observed and forecasted antigenic evolution},
	author = {Park, Soyoon and Widatalla, Talal and Driscoll, Claudia L. and Mille-Fragoso, Luis S. and Specht, Ivan and Gao, Xiaojing and Hie, Brian L.},
	url = {https://doi.org/10.64898/2026.11.10.765734},
	doi = {10.64898/2026.11.10.765734},
	publisher = {bioRxiv},
	year = {2026},
}
```

This pipeline builds on Germinal; please also cite:

```bibtex
@article{mille-fragoso_efficient_2026,
	title = {Efficient generation of epitope-targeted antibodies with Germinal},
	author = {Mille-Fragoso, Luis Santiago and Driscoll, Claudia L. and Wang, John N. and Dai, Haoyu and Widatalla, Talal M. and Zhang, Xiaowe and Hie, Brian L. and Gao, Xiaojing J.},
	url = {https://doi.org/10.1038/s41587-026-03187-0},
	doi = {10.1038/s41587-026-03187-0},
	journal = {Nature Biotechnology},
	year = {2026},
}
```

## Acknowledgments

Germinal builds upon the foundational work of previous hallucination-based protein design pipelines such as ColabDesign and BindCraft and this codebase incorporates code from both repositories. Multi-state inverse folding is performed by the Kuhlman Lab's multi-state fork of ProteinMPNN, vendored as a git submodule at `external/proteinmpnn`, which is the redesign engine for every multi-target configuration shipped here. We are grateful to the developers of these tools for making them available to the research community. 

**Related Work:**
If you use components of this pipeline, please also cite the underlying methods:

- **ColabDesign**: [https://github.com/sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
- **IgLM**: [https://github.com/Graylab/IgLM](https://github.com/Graylab/IgLM)
- **AlphaFold3**: [https://github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)
- **Protenix**: [https://github.com/bytedance/Protenix](https://github.com/bytedance/Protenix)
- **AbMPNN**: [Dreyer, F. A., Cutting, D., Schneider, C., Kenlay, H. & Deane, C. M. Inverse folding for
antibody sequence design using deep learning. (2023).](https://www.biorxiv.org/content/10.1101/2025.05.09.653228v1.full.pdf)
- **ProteinMPNN**: [Dauparas, J. et al. Robust deep learning-based protein sequence design using ProteinMPNN. Science 378, 49-56 (2022).](https://www.science.org/doi/10.1126/science.add2187)
- **ProteinMPNN (Kuhlman Lab multi-state fork)**: [https://github.com/Kuhlman-Lab/proteinmpnn](https://github.com/Kuhlman-Lab/proteinmpnn) — used for multi-state redesign; see [Multi-state design with ProteinMPNN](https://doi.org/10.1101/2023.06.20.545747).
- **PyRosetta**: [https://www.pyrosetta.org/](https://www.pyrosetta.org/)

## License

This repository is licensed under the [Apache License 2.0](LICENSE).

### External Dependencies

Some components require separate licenses that are not included in this repository:

- **IgLM**: Provided under a non-commercial academic license from Johns Hopkins University.  
  See their documentation for details.  
- **PyRosetta**: Provided by the Rosetta Commons and University of Washington under a non-commercial, non-profit license.  
  PyRosetta cannot be redistributed and must be obtained separately.  
  Commercial use requires a separate license. See [https://www.pyrosetta.org](https://www.pyrosetta.org).
