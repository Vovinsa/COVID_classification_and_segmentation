# Covid classification and segmentation

- run to install requirements:

```
python3 install -r COVID_Django/requirements.txt
```

- run to start server:

```
cd COVID_Django
python3 manage.py runserver 0.0.0.0:5000
```

## If you are using Mac on M1

- download [Miniforge](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)
- run to install tensorflow for Mac with M1:

```
python3 -m venv ~/tensorflow-metal
source ~/tensorflow-metal/bin/activate
python -m pip install -U pip
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
conda install -c apple tensorflow-deps
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
```

- run to install requirements:

```
python3 install -r COVID_Django/requirements.txt
```

- run to start server:

```
cd COVID_Django
python3 manage.py runserver 0.0.0.0:5000
```

!["img"](covid.png)
!["img_main"](covid_main.png)
  