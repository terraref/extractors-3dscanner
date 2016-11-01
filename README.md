# 3D scanner extractors

This repository contains extractors that process data originating from:
- 3D Frauhofer top-down height scanner

### PLY to LAS conversion extractor
This extractor converts PLY 3D point cloud files into LAS files. The LAS file will be placed in same directory as PLY file.

_Input_

  - Evaluation is triggered whenever a file is added to a dataset
  - Checks whether the file is a .PLY file
  
_Output_

  - The dataset containing the .PLY file will get a corresponding .LAS file
