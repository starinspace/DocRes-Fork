
<div align=center>

# This is a Fork of DocRes:<br>A Generalist Model Toward Unifying<br>Document Image Restoration Tasks
[original ZZZHANG-jx DocRes](https://github.com/ZZZHANG-jx/DocRes)
</div>

<p align="center">
<img width="981" height="738" alt="bild" src="https://github.com/user-attachments/assets/aeb2ddd6-94c5-4957-b8f6-b544b066a373" />
</p>

# NEW
2026-01-23 Updated the script to fix the Cuda out of Memory.

# TO DO
Add another model support for binarization

## SETUP
```bash
conda create -n docresfork python=3.10.19
conda activate docresfork
cd docresfork
pip install -r requirements.txt
```

1. Put MBD model weights [mbd.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./models/`
2. Put DocRes model weights [docres.pkl](https://1drv.ms/f/s!Ak15mSdV3Wy4iahoKckhDPVP5e2Czw?e=iClwdK) to `./models/`
3. Run the following script and the results will be saved in `./output/`. put your examples in `./input/`.
```bash
python inference.py --im_path ./input/for_dewarping.png --task dewarping --memory_fix 2 --save_dtsprompt 1
```

- `--memory_fix`: fix Cuda out of Memory, use _0_ = No fix (standard), recommended for normal text in big images is _1_, or _2_, for large images with very small text try _3_ = 3000px.
- `--im_path`: the path of input document image
- `--task`: task that need to be executed, it must be one of _binarization_, _dewarping_, _deshadowing_, _appearance_, _deblurring_, or _end2end_
- `--save_dtsprompt`: whether to save the DTSPrompt
  
## CITATION
```
@inproceedings{zhangdocres2024, 
Author = {Jiaxin Zhang, Dezhi Peng, Chongyu Liu , Peirong Zhang and Lianwen Jin}, 
Booktitle = {In Proceedings of the IEEE/CV Conference on Computer Vision and Pattern Recognition}, 
Title = {{DocRes: A Generalist Model Toward Unifying Document Image Restoration Tasks}}, 
Year = {2024}}   
```

```
Fork by = Starinspace
```
