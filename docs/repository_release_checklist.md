# Repository Release Checklist

## Required Before Public Upload

- [ ] Choose and add a `LICENSE`.
- [ ] Confirm that the GitHub repository name is `ReWave-Net`.
- [ ] Change the local `origin` remote from the old `KFAU-Net` repository.
- [ ] Confirm `git status --ignored` shows datasets, checkpoints, and outputs
      as ignored.
- [ ] Confirm no private paths, credentials, tokens, or patient-identifying
      information are present.
- [ ] Confirm that every example MRI image and derived figure can be publicly
      shared under the applicable dataset terms.
- [ ] Run the smoke test and compilation checks.
- [ ] Review the staged diff before committing.

## Recommended Release Files

- [x] Public-facing `README.md`
- [x] Architecture figure
- [x] Method description
- [x] Curated results summary
- [x] Data layout instructions
- [x] Training and evaluation entry points
- [ ] License selected by the repository owner
- [ ] Citation metadata with the final author list
- [ ] Optional downloadable checkpoint hosted in a GitHub Release or external
      artifact store

## Suggested GitHub Settings

- Use `ReWave-Net` as the repository name.
- Add topics such as `mri-reconstruction`, `fastmri`, `wavelet`,
  `deep-learning`, and `unrolled-network`.
- Keep raw fastMRI data out of Git and Git LFS.
- If sharing a checkpoint, attach it to a tagged GitHub Release and document
  its exact training configuration and checksum.
- Enable issue tracking only if you plan to maintain the public repository.

## Suggested First Release

Use a version such as `v0.1.0` for the first research-code release. The release
notes should state:

- the supported task and dataset;
- the exact main experiment configuration;
- the available baselines and missing ablations;
- whether pretrained weights are included; and
- that the repository is research code and not intended for clinical use.
