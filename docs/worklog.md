# sickle
Sickle Prediction


## Work Log 

### <2025-06-28 Sat 08:00-ish> Brandon Li's key changes
Looks like the key change by Brandon Li is using cellpose's remove_edge_masks. This is promising.


### <2025-06-28 Sat 21:01> Using ffmpeg to clip videos on windows
Never done this on Windows. Just did it! 

Steps:

1. download ffmpeg built exe (You should download pre-built exe and not source code)

    Go to https://www.gyan.dev/ffmpeg/builds/

    Download ffmpeg-release-essentials.zip

    Extract it to C:\ffmpeg-7.11
    So that you have ffmpeg.exe under C:\ffmpeg-7.11\bin


2. Add C:\ffmpeg-7.11\bin to the "Path" system environment variable.

   To do this:
   Win + R and run sysadm.cpl
   Adanved -> Environment Variables -> System variables -> Select "Path" -> Edit -> New
   

### <2025-06-28 Sun 23:26> Some findings
1. Plotting masks before and after running remove_edge_masks
   showed no difference. This would explain why She saw no
   difference in results.  But why? Idk what masks is.

2. I found that it's possible to just get (x,y,w,h) bbox
   of every cell. It's possible to filter them probably.

3. Looks like masks has a huge length

   (Pdb) masks
array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint16)
(Pdb) masks.shape
(3648, 5472)
(Pdb)

    Obviously more than the number of cells.

    IDK why.
    But right now I'm just going to save some data
    so that the program debugging can proceed faster


### <2025-07-01 Tue 20:39> Fixing red-blue
{'bbox': (1140, 5, 204, 148), 'class': 0, 'class_prob': 0.9949087500572205, 'state': 1, 'state_prob': 0.999915361404419}

cell_info[cell_id_counter] = {
    'bbox': (x, y, w, h),
    'class': cls_label,
    'class_prob': cls_prob,
    'state': bin_label,
    'state_prob': bin_prob
}

Meaning:
'class': 0 cell class A [one of A B C D E F]
'state': 1 [1 = changed; 0 = unchanged]




















### <2026-09-02 Wed> Folding in the sickling-degree work

Merged the `sickling_degree_classifier` working directory into the repo layout. Notes on
what was surprising rather than what was moved (the moves are in README.md under
"Reorganisation notes").

1. `pipeline_semi_final_detection.py` was loading its classifier from
   `runs/convnext_tiny_sickling_degree_20260505_215821/best_model.pt`. That is training
   scratch — untracked, and the exact path a re-run of `train_vit.py` would overwrite.
   Same hazard the notebooks had with `models/rbc_ckpts/`, one directory over. Hashed it
   against `semi_final_classifier.pt`: byte-identical, so the checkpoint had already been
   promoted and the pipeline just never got repointed. It now loads from `models/`.

2. The three current subtype pipelines import `ViTFeatureExtractor`, which transformers
   dropped in 4.41. Only the incoming semi/final scripts had the
   `ViTImageProcessor` fallback. Ported it across, so `pipelines/` runs on a current
   transformers install.

3. 11,177 `*Zone.Identifier` files, one per downloaded file — an NTFS alternate data
   stream that shows up as a real file on WSL. They were 49% of the file count in the
   directory. Deleted and added to `.gitignore`.

4. `pipeline_semi_final_detection_package.zip` was 452 MB and every one of its six members
   hashed identical to a file already in the tree. Moved to an ignored `dist/`.

5. Morphology features are a dead end on this dataset. 21 hand-computed shape and
   intensity descriptors reach 0.794 macro F1 against the CNN's 0.904, and concatenating
   them with the CNN probabilities (0.902) does not beat the CNN alone. The CNN has
   already learned what those features encode. Worth knowing before anyone tries again.

6. Careful with the two `convnext_tiny` runs. `..._163854` is the analysis baseline —
   every ensemble, threshold, and morphology result scores against it — but `..._215821`
   is the one promoted to `semi_final_classifier.pt`. The gap is 0.0028 macro F1, well
   inside noise, but the checkpoints are different files and easy to mix up.

7. Best result overall is a 3-model probability average (convnext_tiny + vit_b_16 +
   efficientnet_b3) at 0.9074 macro F1, +0.003 over the best single model. Not wired into
   any pipeline: it would mean loading three backbones per frame for a third of a point.
