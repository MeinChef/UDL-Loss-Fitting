from imports import tensorflow as tf
from imports import keras
from imports import numpy as np
from imports import plt
from imports import colors
from imports import skd


# paper for visualising loss surfaces
# https://doi.org/10.48550/arXiv.1712.09913
# article using said article as source
# https://tvsujal.medium.com/visualising-the-loss-landscape-3a7bfa1c6fdf
# another article
# https://mathformachines.com/posts/visualizing-the-loss-landscape/


# TODO: annotate class
class LossSurface(object):
    def __init__(
            self, 
            model, 
            inputs, 
            outputs
        ) -> None:
        
        self.model_ = model
        self.inputs_ = inputs
        self.outputs_ = outputs

    def compile(
            self, 
            range, 
            points, 
            coords
        ) -> None:
        a_grid = tf.linspace(-1.0, 1.0, num = points) ** 3 * range
        b_grid = tf.linspace(-1.0, 1.0, num = points) ** 3 * range
        loss_grid = np.empty([len(a_grid), len(b_grid)])
        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                self.model_.set_weights(coords(a, b))

                loss = self.model_.test_on_batch(
                    self.inputs_, self.outputs_, return_dict=True
                )["loss"]
                loss_grid[j, i] = loss
        self.model_.set_weights(coords.origin_)
        self.a_grid_ = a_grid
        self.b_grid_ = b_grid
        self.loss_grid_ = loss_grid

    def plot(
            self,
            levels = 20, 
            ax = None, 
            **kwargs
        ) -> plt.Axes :

        xs = self.a_grid_
        ys = self.b_grid_
        zs = self.loss_grid_
        if ax is None:
            _, ax = plt.subplots(**kwargs)
            ax.set_title("The Loss Surface")
            ax.set_aspect("equal")
        # Set Levels
        min_loss = zs.min()
        max_loss = zs.max()
        levels = tf.exp(
            tf.linspace(
                tf.math.log(min_loss), 
                tf.math.log(max_loss), 
                num = levels
            )
        )
        # Create Contour Plot
        CS = ax.contour(
            xs,
            ys,
            zs,
            levels = levels,
            cmap = "magma",
            linewidths = 0.75,
            norm = colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
        )
        ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        return ax

class PCACoordinates(object):
    def __init__(self, training_path):
        origin = training_path[-1]
        self.pca_, self.components = get_path_components(training_path)
        self.set_origin(origin)

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc
            for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]

    def set_origin(self, origin, renorm = True):
        self.origin_ = origin
        if renorm:
            self.v0_ = normalize_weights(self.components[0], origin)
            self.v1_ = normalize_weights(self.components[1], origin)


def parameters_to_vector(
        params: keras.Model.weights
    ) -> tf.Tensor:
    """
    Convience function for turning model into a 1D-Tensor containing all variables.
    """
    return tf.concat(
        [tf.reshape(var, [-1]) for var in params],
        axis = 0
        )


def vector_to_parameters(
        vector: tf.Tensor, 
        model: keras.Model
    ) -> None:
    """
    Convenience function for assigning a flat vector to a model's trainable variables.
    """
    
    pointer = 0
    # iterate over variables
    for var in model.get_weights():
        # get the no of numbers that make this variable
        shape = var.shape
        num_params = tf.reduce_prod(shape)
        # take the amount from the vector
        var_vector = vector[pointer : pointer + num_params]
        reshaped = tf.reshape(var_vector, shape)
        var.assign(reshaped)
        # increase the pointer to not access the variables again
        pointer += num_params


def paramlist_to_matrix(
        param_list:list    
    ) -> tf.Tensor:
    
    vec_list = []
    for weights in param_list:
        vec_list.append(parameters_to_vector(weights))
    return tf.stack(
        values = vec_list,
        axis = 1
    )


def shape_weight_matrix_like(
        weight_matrix: tf.Tensor,
        example: list
    ) -> tf.Tensor:
    """
    :param weight_matrix: Tensor of shape (num_params, num_models)
    :type weight_matrix: tf.Tensor
    :param example: the weights of the model, used for shape matching
    :type example: list
    """
    weight_vecs = tf.split(
        weight_matrix, 
        weight_matrix.shape[1], 
        axis = 1
    )

    sizes = [tf.size(v).numpy() for v in example]
    shapes = [v.shape for v in example]
    
    weight_list = []
    for net_weights in weight_vecs:
        # net_weights = tf.reshape(net_weights, [-1])
        # split the long weights vector up into the respective sizes of each layer
        vs = tf.split(net_weights, sizes)
        # and reshape the now correct length vectors into the shape of the weights
        vs = [tf.reshape(v, s) for v, s in zip(vs, shapes)]
        weight_list.append(vs)
    
    return weight_list


def get_path_components(
        training_path, 
        n_components = 2
    ) -> tuple:
    # Vectorize network weights
    weight_matrix = paramlist_to_matrix(training_path)

    # Create components
    pca = skd.PCA(
        n_components = n_components,
        whiten = True
    )
    components = pca.fit_transform(weight_matrix)
    components = tf.convert_to_tensor(
        components,
        dtype = tf.float32
    )
    
    # Reshape to fit network
    example = training_path[0]
    weight_list = shape_weight_matrix_like(components, example)
    
    return pca, weight_list


def normalize_weights(weights, origin):
    norm = []
    for w, wc in zip(weights, origin):
        if tf.is_tensor(w):
            norm.append(
                w * tf.norm(wc) / tf.norm(w)
            )
        elif isinstance(w, np.ndarray):
            norm.append(        
                w * np.linalg.norm(wc) / np.linalg.norm(w)
            )
    return norm


# def non_euklid_transf_1d(
#         alpha: float, 
#         params: tf.Tensor, 
#         param_opt: tf.Tensor
#     ) -> tf.Tensor:
#     return alpha * param_opt + (1 - alpha) * params

# def non_euklid_transf_2d(
#         alpha: float,
#         beta: float,
#         params_opt: tf.Tensor    
#     ) -> tf.Tensor:
#     """
#     Convenience function for transforming the parameters of a network into a useable space.
    
#     :param alpha: Lower bound of the 
#     """
#     a = alpha * params_opt[:,None,None]
#     b = beta * alpha * params_opt[:,None,None]
#     return a + b