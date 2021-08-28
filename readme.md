# Covid classification and segmentation

- run:
*pip3 install requirements.txt*

- run:
*python3 main.py*

## If you are using Mac on M1

- download ["Miniforge"](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
- run:

*python3 -m venv ~/tensorflow-metal*
*source ~/tensorflow-metal/bin/activate*
*python -m pip install -U pip*
*chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh*
*sh ~/Downloads/Miniforge3-MacOSX-arm64.sh*
*source ~/miniforge3/bin/activate*
*conda install -c apple tensorflow-deps*
*python -m pip install tensorflow-macos*
*python -m pip install tensorflow-metal*
  
!["img"](covid.png)
  