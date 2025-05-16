# UDL-Loss-Fitting
Repository for the SummerSemester 2025 Course "Understanding Deep Learning", at University Osnabrück

# Data

We use data collected by the Lower Saxon Ministry for the Environment, Energy and Climate Protection (Niedersächsisches Ministerium für Umwelt, Energie und Klimaschutz). Lower Saxony maintains a network of weather stations to measure air quality (Lufthygienisches Überwachungssystem Niedersachsen) whose most recent data can be downloaded via https://www.umwelt.niedersachsen.de/startseite/themen/luftqualitat/lufthygienische_uberwachung_niedersachsen/aktuelle_messwerte_messwertarchiv/messwertarchiv/download/ . The data we use was obtained by selecting the station "Osnabrück" - not (!) the station "Osnabrück (VS)" - and selecting the components "Luftdruck" (barometric pressure) and "Windrichtung" (wind direction). We selected "Stundenwerte" (hourly measurements) in the timeframe 12.02.2025 through 12.05.2025. The data was downloaded on 13.05.2025 at 12.01 p.m.

# Create environment
You can create the environment needed for this project using:
```
$ conda env create -f env.yml python=3.11
```
After creating the environment, you can activate it using the following command:
```
$ conda activate udl
```
Once activated, you can start the program with
```
$ python src/main.py
```
# Novelty in our Demo
The visualizatition of the data in different graphics that weren't used before by someone else for this task.
