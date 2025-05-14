# UDL-Loss-Fitting
Repository for the SummerSemester 2025 Course "Understanding Deep Learning", at University Osnabrück

# Data

We use data collected by the Lower Saxon Ministry for the Environment, Energy and Climate Protection (Niedersächsisches Ministerium für Umwelt, Energie und Klimaschutz). Lower Saxony maintains a network of weather stations to measure air quality (Lufthygienisches Überwachungssystem Niedersachsen) whose most recent data can be downloaded via https://www.umwelt.niedersachsen.de/startseite/themen/luftqualitat/lufthygienische_uberwachung_niedersachsen/aktuelle_messwerte_messwertarchiv/messwertarchiv/download/ . The data we use was obtained by selecting the station "Osnabrück" - not (!) the station "Osnabrück (VS)" - and selecting the components "Luftdruck" (barometric pressure) and "Windrichtung" (wind direction). We selected "Stundenwerte" (hourly measurements) in the timeframe 12.02.2025 through 12.05.2025. The data was downloaded on 13.05.2025 at 12.01 p.m.


To export a YAML file from an existing environment, type this command in your Terminal: conda my_env export > environment.yml
Here’s an example of what this might look like:

name: udl
channels:
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_gnu
  - bzip2=1.0.8=h4bc722e_7
  - ca-certificates=2025.4.26=hbd8a1cb_0
  - ld_impl_linux-64=2.43=h712a8e2_4
  - libexpat=2.7.0=h5888daf_0
  - libffi=3.4.6=h2dba641_1
  - libgcc=15.1.0=h767d61c_2
  - libgcc-ng=15.1.0=h69a702a_2
  - libgomp=15.1.0=h767d61c_2
  - liblzma=5.8.1=hb9d3cd8_1
  - liblzma-devel=5.8.1=hb9d3cd8_1
  - libnsl=2.0.1=hd590300_0
  - libsqlite=3.49.2=hee588c1_0
  - libuuid=2.38.1=h0b41bf4_0
  - libxcrypt=4.4.36=hd590300_1
  - libzlib=1.3.1=hb9d3cd8_2
  - ncurses=6.5=h2d0b736_3
  - openssl=3.5.0=h7b32b05_1
  - pip=25.1.1=pyh8b19718_0
  - python=3.11.12=h9e4cc4f_0_cpython
  - readline=8.2=h8c095d6_2
  - setuptools=80.1.0=pyhff2d567_0
  - tk=8.6.13=noxft_h4845f30_101
  - tzdata=2025b=h78e105d_0
  - wheel=0.45.1=pyhd8ed1ab_1
  - xz=5.8.1=hbcc6ac9_1
  - xz-gpl-tools=5.8.1=hbcc6ac9_1
  - xz-tools=5.8.1=hb9d3cd8_1
  - pip:
      - absl-py==2.2.2
      - astunparse==1.6.3
      - certifi==2025.4.26
      - charset-normalizer==3.4.2
      - flatbuffers==25.2.10
      - gast==0.6.0
      - google-pasta==0.2.0
      - grpcio==1.71.0
      - h5py==3.13.0
      - idna==3.10
      - keras==3.9.2
      - libclang==18.1.1
      - markdown==3.8
      - markdown-it-py==3.0.0
      - markupsafe==3.0.2
      - mdurl==0.1.2
      - ml-dtypes==0.5.1
      - namex==0.0.9
      - numpy==1.26.4
      - opt-einsum==3.4.0
      - optree==0.15.0
      - packaging==25.0
      - pillow==11.2.1
      - protobuf==5.29.4
      - pygments==2.19.1
      - pyparsing==3.2.3
      - python-dateutil==2.9.0.post0
      - pytz==2025.2
      - requests==2.32.3
      - rich==14.0.0
      - six==1.17.0
      - tensorboard==2.19.0
      - tensorboard-data-server==0.7.2
      - tensorflow==2.19.0
      - tensorflow-io-gcs-filesystem==0.37.1
      - termcolor==3.1.0
      - typing-extensions==4.13.2
      - urllib3==2.4.0
      - werkzeug==3.1.3
      - wrapt==1.17.2

Once you have your YAML file ready, you can create your Conda environment using the following command in your terminal: conda env create -f environment.yml
After creating the environment, you can activate it using the following command: conda activate my_env
Once activated, you can start using the packages in your environment.
