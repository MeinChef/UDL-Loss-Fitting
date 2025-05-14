# UDL-Loss-Fitting
Repository for the SummerSemester 2025 Course "Understanding Deep Learning", at University Osnabrück

# Data

We use data collected by the Lower Saxon Ministry for the Environment, Energy and Climate Protection (Niedersächsisches Ministerium für Umwelt, Energie und Klimaschutz). Lower Saxony maintains a network of weather stations to measure air quality (Lufthygienisches Überwachungssystem Niedersachsen) whose most recent data can be downloaded via https://www.umwelt.niedersachsen.de/startseite/themen/luftqualitat/lufthygienische_uberwachung_niedersachsen/aktuelle_messwerte_messwertarchiv/messwertarchiv/download/ . The data we use was obtained by selecting the station "Osnabrück" - not (!) the station "Osnabrück (VS)" - and selecting the components "Luftdruck" (barometric pressure) and "Windrichtung" (wind direction). We selected "Stundenwerte" (hourly measurements) in the timeframe 12.02.2025 through 12.05.2025. The data was downloaded on 13.05.2025 at 12.01 p.m.


Create a YAML file that lists all the packages and their versions that you want to include in your Conda environment. You can either create a new one or export a YAML file from an existing Conda enviroment. To export a YAML file from an existing environment, type this command in your Terminal: conda my_env export > environment.yml
Here’s an example of what this might look like:

name: my_env
channels:
  - defaults
dependencies:
  - numpy=1.18.1
  - pandas=1.0.1
  - scikit-learn=0.22.1

In this example, the environment is named my_env and includes three packages: numpy, pandas, and scikit-learn with their specific versions
Once you have your YAML file ready, you can create your Conda environment using the following command in your terminal: conda env create -f environment.yml
Replace environment.yml with the path to your YAML file. Conda will then create a new environment based on the specifications in your YAML file.
After creating the environment, you can activate it using the following command: conda activate my_env
Replace my_env with the name of your environment. Once activated, you can start using the packages in your environment.
