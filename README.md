![Python Version](https://img.shields.io/badge/python-3.8--3.11-blue.svg)

# UDL-Loss-Fitting
Repository for the SummerSemester 2025 Course "Understanding Deep Learning", at [University Osnabrück](https://www.uni-osnabrueck.de/)

# Data

We wanted to use data collected by the [Lower Saxon Ministry for the Environment, Energy and Climate Protection](https://www.umwelt.niedersachsen.de/startseite/) (Website in German). Lower Saxony maintains a network of weather stations to measure air quality (Lufthygienisches Überwachungssystem Niedersachsen) whose most recent data can be downloaded [from their website](https://www.umwelt.niedersachsen.de/startseite/themen/luftqualitat/lufthygienische_uberwachung_niedersachsen/aktuelle_messwerte_messwertarchiv/messwertarchiv/download/). The data we use was obtained by selecting the station "Osnabrück" - not the station "Osnabrück (VS)".

Then selecting the components  
- "Luftdruck" (barometric pressure), 
- "Windrichtung" (wind direction), and 
- "Windgeschw." (wind speed).

We selected "Stundenwerte" (hourly measurements) in the timeframe 12.02.2025 through 12.05.2025. The data was downloaded on 13.05.2025 at 00:01.

More about why the data or our model is bad, in [this section](#problems)

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

## von Mises

Since our model tries to predict the direction of wind, which is given on a circle, we are using the von Mises-Fischer Distribution. 
The von Mises Distribution is defined from $-\pi$ to $\pi$ in two dimensions (circle) as follows:

```math
\begin{align}
    f(x|\mu,\kappa) = \frac{\exp(\kappa\cos(x-\mu))}{2\pi I_0(\kappa)}
\end{align}
```

Since we want to predict the angle of the wind, we should focus on $\mu$. Replacing $\mu$ with our model results in:

```math
\begin{align}
    f(x|\mathbf{f\left[x_i, \phi\right]},\kappa) = \frac{\exp(\kappa\cos(x-\mathbf{f\left[x_i, \phi\right]}))}{2\pi I_0(\kappa)}
\end{align}
```

Taking the negative logarithm gives us:


```math
\begin{align}
    L[\mathbf{\phi}] &= \log\left(\frac{\exp(\kappa\cos(x-\mathbf{f\left[x_i, \phi\right]}))}{2\pi I_0(\kappa)}\right) \\
    
    &= \log\exp\left(\kappa\cos(x-\mathbf{f\left[x_i, \phi\right]})\right) - \log(2\pi I_0(\kappa)) \\
    &= \kappa\cos(x-\mathbf{f\left[x_i, \phi\right]}) - \log(2\pi I_0(\kappa))
\end{align}
```

Since the second term does not depend on $\mu$, it can be assumed constant and dropped. That leaves us with the final formula:
```math
\begin{align}
    L[\mathbf{\phi}] = \kappa\cos(x-\mathbf{f\left[x_i, \phi\right]})
\end{align}
```

In [our code](./src/loss.py#L38) this looks like the follows:
```python
@tf.function
def call(self, y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis = -1)
    y_pred = tf.math.l2_normalize(y_pred, axis = -1)

    # 1 - ... because cosine is already between -1 and 1,
    # thus this guarantees positivity.
    return 1 - self.kappa * tf.math.cos(y_true - y_pred)
```


## von Mises-Fischer


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

Not much different to the [von Mises](#von-mises), as per design.
[Code Sippet](./src/loss.py#L75):
```python
@tf.function
def call(self, y_true, y_pred):
    # Normalize to ensure unit vectors
    y_true = tf.math.l2_normalize(y_true, axis = -1)
    y_pred = tf.math.l2_normalize(y_pred, axis = -1)

    # Dot product scaled by kappa
    dot_product = - tf.reduce_sum(y_true * y_pred, axis = -1)
    return self.kappa * dot_product 
```


[Reference for the formulas](https://jstraub.github.io/download/straub2017vonMisesFisherInference.pdf)



## Cosine-Similarity

Another measurement that apparently should work is the Cosine-Similarity measurement. It measures how aligned two vectors are.
This is usually used in embedding spaces for LLMs. We thought we could use it as well, since the angle between two datapoints should be a good measurement on how good our network performs.

Cosine similarity between two vector is defined as follows:

```math
\begin{align}
\cos(\theta) = \frac{A \cdot B}{\Vert A\Vert \Vert B\Vert}
\end{align}
```
Where $\Vert A \Vert$ is the magnitude of the vector.
The [code](./src/loss.py#L109) reads as follows:

```python
@tf.function
def call(self, y_true, y_pred):
    # normalize 
    y_true = tf.norm(y_true, axis = -1)
    y_pred = tf.norm(y_pred, axis = -1)

    return tf.reduce_sum(tf.multiply(y_true, y_pred))
```

## Embedding in Euclidian Space (MSE)

Since we didn't have enough options (or enough working ones, see [Problems](#problems)), [this article](https://medium.com/@john_96423/the-wraparound-problem-predicting-angles-in-machine-learning-44786aa51b91) recommends to transform the angle with sine and cosine and apply the usual Mean Square Error problem on it.
Since this requires now two predictions (one cosine and one sine), the model has to be adjusted to predict two values. This is done by the ["circular" model](./src/model.py#L30).

Noteable is that for the loss the two predictions get their own loss calculation

## Link dump
https://github.com/google-research/vmf_embeddings/blob/main/vmf_embeddings/methods/methods.py


## Euclidian Embedding of Cos, Sin
```python
wind_dir_rad = np.deg2rad(wind_direction)
sin_dir = np.sin(wind_dir_rad)
cos_dir = np.cos(wind_dir_rad)

loss = MSE(pred_sin, true_sin) + MSE(pred_cos, true_cos)
```

# Problems


TODO: 
- make Readme pretty, write something about cosine-similarity
- as well as about sine/cosine loss - and mention that that's the only thing that works
- play around with hyperparams of optimiser
- search internet for sine/cosine stuff
- write about problems we ran into
- look again at cosine similarity, make sure we're predicting speed **AND** direction (vector of dim 1 is stupid)