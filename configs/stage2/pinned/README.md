# Pinned learning rates for Stage 2

Stage 2 does not recalibrate (docs/PREREGISTRATION_STAGE2.md, section 3). These two files are
byte-identical copies of the seed-42 pilot's calibration reports, so every Stage-2 run uses the rates
the pilot selected. Stages read them with `--calibration-report <file> --calibration-report-sha256 <hash>`
and refuse a file whose hash differs.

| File | Pilot source (`runs/<pilot run id>/`) | sha256 | Selected |
|---|---|---|---|
| `contamination_MR_calibration_report.json` | `calibrate_MR/calibration_report.json` | `c4b5d773aa439c12c3deb83b16be8d3abed9109ec0d7844f4d883ca17b0b0cf7` | 2e-5 (MR arm revision 3, arm fingerprint `27499cbc…`) |
| `unlearning_U3_calibration_report.json` | `calibrate_unlearning_U3/calibration_report.json` | `8cbcf00cd15d72de4c13da5ebac7fb40e2e84b3835a6b45f0c2dd38af760da83` | GA 2e-6, NPO 2e-6 (revision 3, fingerprint `272637f8…`) |

The pilot run id is `pilot-medical-s0-s50-s100-qwen05b__qwen-qwen2-5-0-5b__seed42__synthetic-patient-f-v2__cfg2396728f114a`.

These are the exact reports the pilot used: the pilot's matched C_i manifests record
`calibration_report_sha256 = c4b5d773…`, and its revision-3 trajectory plans record `8cbcf00c…`
(verified 2026-11-10). Never edit these files. A different rate is an amendment with new copies under
new names.
