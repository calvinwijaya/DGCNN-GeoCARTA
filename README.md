1. `05_predict.py`: directly classify point cloud data into 2 classes
2. `05_predict_csf`: use CSF to classify ground and off-ground, then off-ground classified into 2 classes, total 3 classes (1 from CSF, 2 from DGCNN). In this version, CSF is done first then split per blocks to classify 2 classes.
3. `05_predict_csf2`: also use CSF to classify ground and off-ground, then off-ground classified into 2 classes, total 3 classes (1 from CSF, 2 from DGCNN). In this version, point cloud data is splitted per blocks first then do CSF for each blocks
4. `05_predict_csf2_lts`: Same with `05_predict_csf2`, but if `05_predict_csf2` merge data into 1 then export, in this version, each block is exported.
5. `05_predict_3class`: directly classify point cloud data into 3 classes: ground(2), vegetation(5), building(6)
