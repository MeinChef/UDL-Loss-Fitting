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
$ python src/main.py [FLAGS]
```

## Flags
### `--model`
A flag for selecting the model to train on. Different flags also have an impact on how the data is prepared, as the LSTM models (lstm, circular) predict on sequences. The sequence length should be define in `cfg/cfg.yml` with the key "seq_len".\
Options:
- `dense` - A densly connected, strictly feed-forward model. Layers are as follows: 
![Picture of Network structure](./img/dense.png)
- `lstm` - An LSTM network, designed to predict the direction of the wind using the last `seq_len` datapoints. Structure is as follows:
# TODO update picture 
![Picture of LSTM Network structur](./img/lstm.png)
- `circular` - The model structure is the same as the LSTM one, with the only difference that the last layer has two output neurons. This model is exclusively to be used with loss `mse`. The angle is embedded in euclidian space by transforming it with $\sin$ and $\cos$. 

# TODO update loss.py and remove vMF, cosine, change circular to sine_cosine
### `--loss`: 
This flag is selecting the loss to be used during training and testing. The different losses are explained in section [Losses](#explanation-of-losses).

Options:
- `mse` - The Mean Squared Error, as used in many state of the art networks. [Caveats and adjustments](#embedding-in-euclidian-space-mse)
- `vM` - A loss based on the von Mises distribution. [Explanation and derivation](#von-mises)

# TODO nuke this
### `--problem`:


# Novelty in our Demo
<!-- The visualizatition of the data in different graphics that weren't used b`--model`:
efore by someone else for this task. -->
The visualisation of the loss surface is done in a way that is not described in papers already.

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
What I only realised after implementing that and trying to getting it to work, is that for having that working, a vector needs to be at least 2D. Which is not what we are predicting. We are predicting a singular value.
And after searching on how to circumevent that problem, I found the following:
[Angle to Vector](https://math.stackexchange.com/questions/180874/convert-angle-radians-to-a-heading-vector) 
And that directly leads to: 

## Embedding in Euclidian Space (MSE)

[This StackExchange post](https://math.stackexchange.com/questions/180874/convert-angle-radians-to-a-heading-vector) talks about transforming an angle with sine and cosine to a headings vector on the unit circle. On these two values the Mean Square Error can be applied to it. 
The same approach is also mentioned in [this article](https://medium.com/@john_96423/the-wraparound-problem-predicting-angles-in-machine-learning-44786aa51b91)

Since this requires now two predictions (one cosine and one sine), the model has to be adjusted to predict two values. This is done by the ["circular" model](./src/model.py#L30).

Noteable is that for the loss the two predictions get their own loss calculation respectively, which then get added.
Per default the implementation in tensorflow for MSE is taking the mean of the -1st axis. But since our prediction is of the shape [batch_size, 2], we would be taking the mean of the sine/cosine.
Our implementation uses the axis 0 instead, taking the mean of the batch and adding column one and two together.

[The implementation](./src/loss.py#L131) is as follows:

```python
@tf.function
def call(self, y_true, y_pred):
    return tf.math.reduce_sum(
        tf.math.reduce_mean(
            tf.math.square(y_true - y_pred),
            axis = self.axis
        ),
        axis = -1
    )
```

# Problems

Deep learning in itself is an optimsation problem of utmost complexity. In our problem we never got the network to remotely predict the data. Many different approaches were tried, using different losses and network structures.

Often the test predictions looked like this (datapoints have been numbered `1` to `n`, since they don't have a y value anymore):

![Bad test predictions, with the network predicting the same value for every datpoint](./img/lstm_150ep.png) 

The loss during that drop off fast during the first epoch, but remained constant during the rest of the training.
![Loss of the first five epochs, with a drop of loss from the first to the second epoch](./img/loss-lstm-5.png)
![Loss of all epochs](./img/loss-lstm-150.png)


The only thing we could remotely call success were using the LSTM model with the [sine/cosine embedding](#embedding-in-euclidian-space-mse):

![Decent test predictions, with the predictions spread](./img/circ_150ep_cossin.png)

The loss during this run also showed some significant improvements over the above run.

![Loss of the LSTM model with the sine/cosine embedding](./img/loss-circ-150.png)

TODO: 

- play around with hyperparams of optimiser
