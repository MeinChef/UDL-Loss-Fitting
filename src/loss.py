from imports import tensorflow as tf
# from imports import tensorflow_probability as tfp

# https://doi.org/10.48550/arXiv.2103.15718
# mean absolute error?
# just cosine similarity?

def angle_cosine_loss(y_true, y_pred):
        y_true = tf.math.l2_normalize(y_true, axis=1)
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        return 1 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

def von_Mises(y_true, y_pred):
    """
    Custom loss function for von Mises distribution.
    :param y_true: true values (angles in radians)
    :type y_true: tf.Tensor

    :param y_pred: predicted values (angles in radians)
    :type y_pred: tf.Tensor
    
    :return: computed loss
    :rtype: tf.Tensor
    """
    # Convert angles to unit vectors
    y_true_vector = tf.stack([tf.cos(y_true), tf.sin(y_true)], axis=-1)
    y_pred_vector = tf.stack([tf.cos(y_pred), tf.sin(y_pred)], axis=-1)

    # Calculate the cosine similarity
    cosine_similarity = tf.reduce_sum(y_true_vector * y_pred_vector, axis=-1)

    # Calculate the von Mises loss
    loss = -tf.math.log(cosine_similarity + 1e-10)  # Add small constant to avoid log(0)

    return tf.reduce_mean(loss)


def von_mises_loss_fixed_kappa(kappa=1.0):
    """
    Creates a von Mises loss function with fixed kappa.
    
    Args:
        kappa: concentration parameter (must be > 0)
        
    Returns:
        A loss function that can be used in model.compile()
    """
    def loss_fn(y_true, y_pred):
        """
        y_true: shape [batch_size, 1] — true angles (in radians)
        y_pred: shape [batch_size, 1] — predicted mu (mean direction)
        """
        mu = y_pred[:, 0]
        theta = y_true[:, 0]

        vm = tfp.distributions.VonMises(loc=mu, concentration=kappa)
        nll = -vm.log_prob(theta)  # negative log likelihood

        return tf.reduce_mean(nll)

    return loss_fn