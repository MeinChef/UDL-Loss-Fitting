from imports import tensorflow as tf
from imports import keras

# https://doi.org/10.48550/arXiv.2103.15718
# mean absolute error?
# just cosine similarity?

class VonMisesFisher(keras.losses.Loss):
    def __init__(
            self, 
            kappa: float = 1.0,
            reduction = "sum_over_batch_size",
            name = "von_mises_fisher",
            dtype = tf.float32
        ):
        """
        Custom loss for von Mises-Fisher distribution, predicting $\mu$.
        The intuition behind kappa is that bigger values penalize angle deviations more.
        
        :param kappa: concentration parameter (must be > 0)
        :type kappa: float

        :param reduction: type of reduction to apply to the loss
        :type reduction: tf.keras.losses.Reduction

        :param name: name of the loss function
        :type name: str
        """

        super().__init__(
            reduction = reduction, 
            name = name
            )
        self.kappa = tf.convert_to_tensor(kappa, dtype = dtype)

    @tf.function
    def call(self, y_true, y_pred):
        # Normalize to ensure unit vectors
        y_true = tf.math.l2_normalize(y_true, axis = -1)
        y_pred = tf.math.l2_normalize(y_pred, axis = -1)

        # Dot product scaled by kappa
        dot_product = tf.reduce_sum(y_true * y_pred, axis = -1)
        loss = -self.kappa * dot_product
        return loss  # shape: (batch,)
