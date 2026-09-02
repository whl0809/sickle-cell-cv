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


















