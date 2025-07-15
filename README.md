![Python Version](https://img.shields.io/badge/python-3.8--3.11-blue.svg)

# UDL-Loss-Fitting
Repository for the SummerSemester 2025 Course "Understanding Deep Learning", at [University Osnabrück](https://www.uni-osnabrueck.de/)

# Data

We use data collected by the [Lower Saxon Ministry for the Environment, Energy and Climate Protection](https://www.umwelt.niedersachsen.de/startseite/) (Website in German). Lower Saxony maintains a network of weather stations to measure air quality (Lufthygienisches Überwachungssystem Niedersachsen) whose most recent data can be downloaded [from their website](https://www.umwelt.niedersachsen.de/startseite/themen/luftqualitat/lufthygienische_uberwachung_niedersachsen/aktuelle_messwerte_messwertarchiv/messwertarchiv/download/). The data we use was obtained by selecting the station "Osnabrück" - not the station "Osnabrück (VS)".

Then selecting the components  
- "Luftdruck" (barometric pressure), 
- "Windrichtung" (wind direction), and 
- "Windgeschw." (wind speed).

We selected "Stundenwerte" (hourly measurements) in the timeframe 12.02.2025 through 12.05.2025. The data was downloaded on 13.05.2025 at 00:01.

# Quickstart
## Create environment
You can create the environment needed for this project using:
```
$ conda env create -f env.yml python=3.11
```
After creating the environment, you can activate it using the following command:
```
$ conda activate udl
```
## Execute Program
Once activated, navigate to the folder of the repository. Then you can execute the program using
```
$ python src/main.py
```

# Novelty in our Demo
The visualizatition of the data in different graphics that weren't used before by someone else for this task.

# Explanation of Losses
## Based on von Mises-Fischer Distribution
Since our model tries to predict the direction of wind, which is given on a circle, we are using the von Mises-Fischer Distribution. 
The von Mises Distribution is defined from $-\pi$ to $\pi$ in two dimensions (circle) as follows:

```math
f(x|\mu,\kappa) = \frac{\exp(\kappa\cos(x-\mu))}{2\pi I_0(\kappa)}
```

But since we are having three dimensions (speed, direction, pressure), a variant of the von Mises distribution, the _von Mises-Fischer Distribution_ that modifies the distribution to $p$ dimensions, could also be used. 

```math
\begin{align}
    f_p (y|\mu,\kappa) &= C_p(\kappa)\exp\left(\kappa \mu^\top y\right)  \\
    C_p(\kappa) &= \frac{\kappa^{p/2-1}}{(2\pi)^{p/2}I_{p/2-1}(\kappa)} 
\end{align}
```
The modified version for three dimensions is defined as:

```math
\begin{align}
    f_p (y|\mu,\kappa) = \frac{\kappa\exp(\kappa(\mu^\top y-1))}{2\pi(1-\exp(-2\kappa))}
\end{align}
```

Let us now replace the variable $\mu$ with our model:

```math
\begin{align}
    f_p (y|\mathbf{f\left[x_i, \phi\right]},\kappa) = \frac{\kappa\exp(\kappa(\mathbf{f\left[x_i, \phi\right]}^\top y-1))}{2\pi(1-\exp(-2\kappa))}
\end{align}
```
we assume $\kappa$ to be an unknown constant. 

Constructing the negative Log-Likelihood with $i$ being an individual datapoint, and $I$ the dataset.

```math
\begin{align}
    L[\mathbf{\phi}] &= - \sum_{i=1}^{I} \log \left[ Pr(y_i|\mathbf{f\left[x_i, \phi\right]}), \kappa \right] \\
    &= - \sum_{i=1}^{I} \log \left[\frac{\kappa\exp\left(\kappa\left(\mathbf{f\left[x_i, \phi\right]}^\top y_i-1\right)\right)}{2\pi(1-\exp(-2\kappa))} \right]  \\
    &= - \sum_{i=1}^{I} \left[\log\left(\kappa\exp(\kappa(\mathbf{f\left[x_i, \phi\right]}^\top y_i-1))\right) - \log \left(2\pi(1-\exp(-2\kappa))\right)\right]  \\
    &= - \sum_{i=1}^{I} \left[\log\kappa + \log\left(\exp(\kappa(\mathbf{f\left[x_i, \phi\right]}^\top y_i-1))\right) - \log \left(2\pi(1-\exp(-2\kappa))\right) \right] \\
    &= - \sum_{i=1}^{I} \left[ \log\kappa + \kappa y_i\mathbf{f\left[x_i, \phi\right]}^\top - \log \left(2\pi(1-\exp(-2\kappa))\right)\right] \\
    &= -\left(N\log(\kappa) + \kappa \sum_{i=1}^{I} \mathbf{f\left[x_i, \phi\right]}^\top  y_i -\log\left(2\pi(1-\exp(-2\kappa))\right) \right)
\end{align}
```
Dropping everything that does not depend on $\mu$ results in:

```math
\begin{align}
    L[\mathbf{\phi}] &= - \kappa \sum_{i=1}^{I} \mathbf{f[x_i,\phi]}^\top y_i 
\end{align}
```

[Reference for the formulas](https://jstraub.github.io/download/straub2017vonMisesFisherInference.pdf)

## Cosine-Similarity



## Link dump
https://github.com/google-research/vmf_embeddings/blob/main/vmf_embeddings/methods/methods.py