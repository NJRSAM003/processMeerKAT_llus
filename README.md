<p align="center">
   <img src="https://raw.githubusercontent.com/idia-pipelines/idia-pipelines.github.io/master/assets/idia_logo.jpg" alt="IDIA pipelines"/>
</p>

# processMeerKAT_llus — MeerKAT Pipeline (fork)

This is a personal fork of the [IDIA MeerKAT pipeline](https://github.com/idia-astro/pipelines), a radio interferometric calibration pipeline designed to process MeerKAT data. It implements cross-calibration, self-calibration, and science imaging. This fork tracks the upstream `master` and adds a number of changes aimed at **full-polarization** processing, **Python 3.12** compatibility, and **science imaging** improvements.

### What this fork changes relative to upstream `master`

* **Polarization calibration on linear feeds** — L-band polarization calibrator support in `setjy`, plus XY-phase ambiguity solving when the polarization calibrator and phase calibrator share a name.
* **`polcalfield` config option** (`[crosscal]`) — explicit fallback XY-phase calibrator, only used when no canonical pol calibrator (3C286/3C138/3C48/J1130-1449) is found in the MS. Optional; defaults to `''` and is auto-annotated by `-B`.
* **`atrous_do` config option** (`[selfcal]`) — enables PyBDSF à-trous (wavelet) decomposition during self-cal source finding to better recover extended/diffuse emission. Defaults to `False` (existing behaviour). Applied in `selfcal_part2.py`.
* **Science-imaging masking modes** (`[image]`) — choose `usemask = 'user'` (standard, uses `mask`) or `usemask = 'auto-multithresh'` (uses `sidelobethreshold`, `noisethreshold`, `lownoisethreshold`, `negativethreshold` instead).
* **PyBDSF-driven spectral-index (alpha) imaging** — for multi-Stokes / non-`I` `mtmfs` runs, the science imaging step builds a noise-thresholded `alpha` map and `alpha.error` map (with a restoring beam inherited from Stokes I so PyBDSF can read it), controlled by `alpha_nsigma`.
* **Per-SPW science imaging** (`[image]`) — set `spw_cube = True` to image each spectral window separately (into `SPWs_full_stokes/`) instead of producing a single full-bandwidth averaged image. Frequency labels are auto-derived from the MS metadata, and `spwid` optionally restricts which SPWs are imaged (`''` = all). Combine with `stokes = 'IQUV'` for full-Stokes per-SPW imaging.
* **Automatic log cleanup** — once all pipeline jobs finish, a lightweight dependent SLURM job removes stray `casa*.log` files from the working directory.
* **Python 3.12 fixes** — `SafeConfigParser` → `RawConfigParser`, invalid escape-sequence `SyntaxWarning`s resolved.

## Requirements

This pipeline is designed to run on the Ilifu cluster, making use of SLURM and MPICASA. For other uses, please contact the authors. Currently, use of the pipeline requires access to the Ilifu cloud infrastructure. You can request access using the following [form](http://docs.ilifu.ac.za/#/getting_started/request_access).

## Quick Start

**Note: It is not necessary to copy the raw data (i.e. the MS) to your working directory. The first step of the pipeline does this for you by creating an MMS or MS, and does not attempt to manipulate the raw data (e.g. stored in `/idia/projects` - see [data format](https://idia-pipelines.github.io/docs/processMeerKAT/Example-Use-Cases/#data-format)).**

## 1. Setup the pipeline in your environment

In order to use the `processMeerKAT.py` script, source this fork's `setup.sh` on [ilifu](https://docs.ilifu.ac.za/#/):

        source /users/amani/processMeerKAT_fork/processMeerKAT/setup.sh

This adds the correct paths to your `$PATH` and `$PYTHONPATH` to use the pipeline. You could consider adding this to your `~/.profile` or `~/.bashrc` for future use.

> If you switch between this fork and the upstream install (`source /idia/software/pipelines/master/setup.sh`), re-source the one you want and regenerate your sbatch scripts (`-R`) so they point at the correct `processMeerKAT` directory.

### 2. Build a config file:

#### a. For continuum/spectral line processing :

        processMeerKAT.py -B -C myconfig.txt -M mydata.ms

#### b. For polarization processing :

        processMeerKAT.py -B -C myconfig.txt -M mydata.ms -P

#### c. Including self-calibration :

        processMeerKAT.py -B -C myconfig.txt -M mydata.ms -2

#### d. Including science imaging :

        processMeerKAT.py -B -C myconfig.txt -M mydata.ms -I

This defines several variables that are read by the pipeline while calibrating the data, as well as requesting resources on the cluster. The [config file parameters](https://idia-pipelines.github.io/docs/processMeerKAT/config-files) are described by in-line comments in the config file itself wherever possible. The `[-P --dopol]` option can be used in conjunction with the `[-2 --do2GC]` and `[-I --science_image]` options to enable polarization calibration as well as [self-calibration](https://idia-pipelines.github.io/docs/processMeerKAT/self-calibration-in-processmeerkat) and [science imaging](https://idia-pipelines.github.io/docs/processMeerKAT/science-imaging-in-processmeerkat).

### 3. To run the pipeline:

        processMeerKAT.py -R -C myconfig.txt

This will create `submit_pipeline.sh`, which you can then run with `./submit_pipeline.sh` to submit all pipeline jobs to the SLURM queue. After all jobs complete, stray `casa*.log` files are automatically removed from the working directory.

Other convenience scripts are also created that allow you to monitor and (if necessary) kill the jobs.

* `summary.sh` provides a brief overview of the status of the jobs in the pipeline
* `findErrors.sh` checks the log files for commonly reported errors (after the jobs have run)
* `killJobs.sh` kills all the jobs from the current run of the pipeline, ignoring any other (unrelated) jobs you might have running.
* `cleanup.sh` wipes all the intermediate data products created by the pipeline. This is intended to be launched after the pipeline has run and the output is verified to be good.

For help, run `processMeerKAT.py -h`, which provides a brief description of all the [command line options](https://idia-pipelines.github.io/docs/processMeerKAT/using-the-pipeline#command-line-options).

## Fork-specific config keys

These keys are added/used by this fork. They all have sensible defaults, so existing config files keep working unchanged.

| Section | Key | Default | Purpose |
| --- | --- | --- | --- |
| `[crosscal]` | `polcalfield` | `''` | Fallback XY-phase calibrator; only used when no canonical pol calibrator is in the MS. |
| `[selfcal]` | `atrous_do` | `False` | Enable PyBDSF à-trous (wavelet) decomposition during self-cal source finding. |
| `[image]` | `usemask` | `'user'` | `'user'` uses `mask`; `'auto-multithresh'` uses the thresholds below instead. |
| `[image]` | `sidelobethreshold` | `0.5` | Only used when `usemask = 'auto-multithresh'`. |
| `[image]` | `noisethreshold` | `5.0` | Only used when `usemask = 'auto-multithresh'`. |
| `[image]` | `lownoisethreshold` | `0.01` | Only used when `usemask = 'auto-multithresh'`. |
| `[image]` | `negativethreshold` | `0.0` | Only used when `usemask = 'auto-multithresh'`. |
| `[image]` | `alpha_nsigma` | `1.0` | Sigma cut for the final alpha mask (used when `stokes != 'I'` to produce a spectral-index image). |
| `[image]` | `spw_cube` | `False` | Image each SPW separately into `SPWs_full_stokes/` instead of one full-bandwidth averaged image. |
| `[image]` | `spwid` | `''` | Comma-separated SPW IDs to image when `spw_cube = True` (e.g. `'0,1,2'`); `''` = all SPWs. |

## Using multiple spectral windows (new in v1.1)

Starting with v1.1 of the processMeerKAT pipeline, the default behaviour is to split up the MeerKAT band into several spectral windows (SPWs), and process each concurrently. This results in a few major usability changes as outlined below:

1. **Calibration output** : Since the calibration is performed independently per SPW, all the output specific to that SPW is within its own directory. Output such as the calibration tables, logs, plots etc. per SPW can be found within each SPW directory.

2. **Logs in the top level directory** : Logs in the top level directory (*i.e.,* the directory where the pipeline was launched) correspond to the scripts in the `precal_scripts` and `postcal_scripts` variables in the config file. These scripts are run from the top level before and after calibration respectively. By default these correspond to the scripts to calculate the reference antenna (if enabled), partition the data into SPWs, and concat the individual SPWs back into a single MS/MMS.

More detailed information about SPW splitting is found [here](https://idia-pipelines.github.io/docs/processMeerKAT/config-files#spw-splitting).

The documentation can be accessed on the [pipelines website](https://idia-pipelines.github.io/docs/processMeerKAT), or on the [Github wiki](https://github.com/idia-astro/pipelines/wiki).
