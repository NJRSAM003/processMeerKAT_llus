<p align="center">
   <img src="https://raw.githubusercontent.com/idia-pipelines/idia-pipelines.github.io/master/assets/idia_logo.jpg" alt="IDIA pipelines"/>
</p>

# processMeerKAT_llus — MeerKAT Pipeline (fork)

This is a personal fork of the [IDIA MeerKAT pipeline](https://github.com/idia-astro/pipelines), a radio interferometric calibration pipeline designed to process MeerKAT data. It implements cross-calibration, self-calibration, and science imaging. This fork tracks the upstream `master` and adds a number of changes aimed at **full-polarization** processing, **Python 3.12** compatibility, and **science imaging** improvements.

### What this fork changes

* **Polarization calibration on linear feeds** — L-band polarization calibrator support in `setjy`, plus XY-phase ambiguity solving when the polarization calibrator and phase calibrator share a name.
* **`polcalfield` config option** (`[crosscal]`) — explicit fallback XY-phase calibrator, only used when no canonical pol calibrator (3C286/3C138/3C48/J1130-1449) is found in the MS. Optional; defaults to `''` and is auto-annotated by `-B`.
* **`atrous_do` config option** (`[selfcal]`) — enables PyBDSF à-trous (wavelet) decomposition during self-cal source finding to better recover extended/diffuse emission. Defaults to `False` (existing behaviour). Applied in `selfcal_part2.py`.
* **Science-imaging masking modes** (`[image]`) — choose `usemask = 'user'` (standard, uses `mask`) or `usemask = 'auto-multithresh'` (uses `sidelobethreshold`, `noisethreshold`, `lownoisethreshold`, `negativethreshold` instead).
* **PyBDSF-driven spectral-index (alpha) imaging** — for multi-Stokes / non-`I` `mtmfs` runs, the science imaging step builds a noise-thresholded `alpha` map and `alpha.error` map (with a restoring beam inherited from Stokes I so PyBDSF can read it), controlled by `alpha_nsigma`.
* **Radio-continuum cube imaging** (`[image]`) — set `spw_cube = True` to image each spectral window separately (into `SPW_MFSs/`) and then merge them into a single 4D (RA, Dec, Stokes, frequency) cube instead of one full-bandwidth averaged image. Each SPW keeps its own restoring beam, so the cube is written with a per-plane CASA beam table (the correct, frequency-dependent beam on every channel). Frequency labels are auto-derived from the MS metadata, `spwid` optionally restricts which SPWs are imaged (`''` = all), and combining with `stokes = 'IQUV'` gives a full-Stokes continuum cube.
* **Automatic log cleanup** — once all pipeline jobs finish, a lightweight dependent SLURM job moves stray `casa*.log` files from the working directory into the `logs/` folder.
* **Python 3.12 fixes** — `SafeConfigParser` → `RawConfigParser`, invalid escape-sequence `SyntaxWarning`s resolved.

## Requirements

This pipeline is designed to run on the Ilifu cluster, making use of SLURM and MPICASA. For other uses, please contact the authors. Currently, use of the pipeline requires access to the Ilifu cloud infrastructure. You can request access using the following [form](http://docs.ilifu.ac.za/#/getting_started/request_access).

## Quick Start

**Note: It is not necessary to copy the raw data (i.e. the MS) to your working directory. The first step of the pipeline does this for you by creating an MMS or MS, and does not attempt to manipulate the raw data (e.g. stored in `/idia/projects` - see [data format](https://idia-pipelines.github.io/docs/processMeerKAT/Example-Use-Cases/#data-format)).**

## 1. Setup the pipeline in your environment

First clone this fork somewhere on [ilifu](https://docs.ilifu.ac.za/#/), then source its `setup.sh` to use the `processMeerKAT.py` script:

        git clone https://github.com/NJRSAM003/processMeerKAT_llus.git
        source /path/to/processMeerKAT_llus/processMeerKAT/setup.sh

Replace `/path/to/processMeerKAT_llus` with wherever you cloned the fork. This adds the correct paths to your `$PATH` and `$PYTHONPATH` to use the pipeline. You could consider adding this to your `~/.profile` or `~/.bashrc` for future use.

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

This will create `submit_pipeline.sh`, which you can then run with `./submit_pipeline.sh` to submit all pipeline jobs to the SLURM queue. After all jobs complete, stray `casa*.log` files are automatically moved from the working directory into the `logs/` folder.

Other convenience scripts are also created that allow you to monitor and (if necessary) kill the jobs.

* `summary.sh` provides a brief overview of the status of the jobs in the pipeline
* `findErrors.sh` checks the log files for commonly reported errors (after the jobs have run)
* `killJobs.sh` kills all the jobs from the current run of the pipeline, ignoring any other (unrelated) jobs you might have running.
* `cleanup.sh` wipes all the intermediate data products created by the pipeline. This is intended to be launched after the pipeline has run and the output is verified to be good.

For help, run `processMeerKAT.py -h`, which provides a brief description of all the [command line options](https://idia-pipelines.github.io/docs/processMeerKAT/using-the-pipeline#command-line-options).

## Fork-specific config keys

These keys are added/used by this fork. They all have sensible defaults, so existing config files keep working unchanged.

<table>
  <thead>
    <tr><th>Section</th><th>Key</th><th>Default</th><th>Purpose</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><code>[crosscal]</code></td>
      <td><code>polcalfield</code></td>
      <td><code>''</code></td>
      <td>Fallback XY-phase calibrator; only used when no canonical pol calibrator is in the MS.</td>
    </tr>
    <tr>
      <td><code>[selfcal]</code></td>
      <td><code>atrous_do</code></td>
      <td><code>False</code></td>
      <td>Enable PyBDSF à-trous (wavelet) decomposition during self-cal source finding.</td>
    </tr>
    <tr>
      <td rowspan="8"><code>[image]</code></td>
      <td><code>usemask</code></td>
      <td><code>'user'</code></td>
      <td><code>'user'</code> uses <code>mask</code>; <code>'auto-multithresh'</code> uses the thresholds below instead.</td>
    </tr>
    <tr>
      <td><code>sidelobethreshold</code></td>
      <td><code>0.5</code></td>
      <td rowspan="4">Only used when <code>usemask = 'auto-multithresh'</code>.</td>
    </tr>
    <tr>
      <td><code>noisethreshold</code></td>
      <td><code>5.0</code></td>
    </tr>
    <tr>
      <td><code>lownoisethreshold</code></td>
      <td><code>0.01</code></td>
    </tr>
    <tr>
      <td><code>negativethreshold</code></td>
      <td><code>0.0</code></td>
    </tr>
    <tr>
      <td><code>alpha_nsigma</code></td>
      <td><code>1.0</code></td>
      <td>Sigma cut for the final alpha mask (used when <code>stokes != 'I'</code> to produce a spectral-index image).</td>
    </tr>
    <tr>
      <td><code>spw_cube</code></td>
      <td><code>False</code></td>
      <td>Image each SPW separately into <code>SPW_MFSs/</code> and merge them into a single 4D (RA, Dec, Stokes, freq) cube with a per-plane beam table, instead of one full-bandwidth averaged image.</td>
    </tr>
    <tr>
      <td><code>spwid</code></td>
      <td><code>''</code></td>
      <td>Comma-separated SPW IDs to image when <code>spw_cube = True</code> (e.g. <code>'0,1,2'</code>); <code>''</code> = all SPWs.</td>
    </tr>
  </tbody>
</table>

## Documentation

By default the pipeline splits the MeerKAT band into several spectral windows (SPWs) and processes each concurrently — this fork keeps that upstream behaviour unchanged. See [SPW splitting](https://idia-pipelines.github.io/docs/processMeerKAT/config-files#spw-splitting) for the details.

Full documentation is on the [pipelines website](https://idia-pipelines.github.io/docs/processMeerKAT) and the [Github wiki](https://github.com/idia-astro/pipelines/wiki).
