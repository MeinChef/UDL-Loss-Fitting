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
            model: keras.Model, 
            inputs: tf.Tensor, 
            outputs: tf.Tensor
        ) -> None:
        
        self.model_ = model
        self.inputs_ = inputs
        self.outputs_ = outputs

    def compile(
            self, 
            range: float, 
            points: int, 
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
            levels: int = 20, 
            ax: plt.Axes = None, 
            **kwargs
        ) -> plt.Axes :

        xs = self.a_grid_
        ys = self.b_grid_
        zs = self.loss_grid_

        # since especially with the von Mises loss, the values of this array might be negative
        # substracting the minimum value from this array preserves the relationship between the
        # values, while the minimum becomes 0. Adding a small value (0.001) just shifts the whole thing 
        # to >0 (important when applying log)
        zs = zs - zs.min() + 0.001
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
            norm = colors.LogNorm(vmin=min_loss, vmax=max_loss),
        )
        ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        return ax

class PCACoordinates(object):
    def __init__(
            self, 
            training_path: list[tf.Tensor]
        ) -> None:
        origin = training_path[-1]
        self.pca_, self.components = get_path_components(training_path)
        self.set_origin(origin)

    def __call__(
            self, 
            a: float, 
            b: float
        ) -> tf.Tensor:
        return [
            a * w0 + b * w1 + wc
            for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]

    def set_origin(self,
            origin: list[tf.Tensor], 
            renorm: bool = True
        ) -> None:
        self.origin_ = origin
        if renorm:
            self.v0_ = normalize_weights(self.components[0], origin)
            self.v1_ = normalize_weights(self.components[1], origin)


def parameters_to_vector(
        params: list[np.ndarray | tf.Tensor]
    ) -> tf.Tensor:
    """
    Convience function for turning model into a 1D-Tensor containing all variables.
    """
    return tf.concat(
        [tf.reshape(var, [-1]) for var in params],
        axis = 0
        )

def paramlist_to_matrix(
        param_list:list    
    ) -> tf.Tensor:
    """
    Convenience function to convert a list of model weights into a matrix.
    :param param_list: List of model weights.
    :type param_list: list[tf.Tensor]
    :return: Tensor of shape (num_params, num_models)
    :rtype: tf.Tensor
    """
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
    Convenience function to reshape a weight matrix to the shape of the model's weights.
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

    """
    Convenience function to get the PCA components of a training path.
    :param training_path: List of model weights.
    :type training_path: list[tf.Tensor]
    :param n_components: Number of components to return.
    :type n_components: int
    :return: PCA object and components.
    :rtype: tuple[sklearn.decomposition.PCA, tf.Tensor]
    """

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

def weights_to_coordinates(
        coords: PCACoordinates, 
        training_path: list[tf.Tensor]
    ) -> tf.Tensor:
    """Project the training path onto the first two principal components
using the pseudoinverse."""
    components = [coords.v0_, coords.v1_]
    comp_matrix = paramlist_to_matrix(components)
    # the pseudoinverse
    comp_matrix = tf.linalg.pinv(comp_matrix)
    # the origin vector
    w_c = parameters_to_vector(training_path[-1])
    # center the weights on the training path and project onto components
    coord_path = []
    for weights in training_path:
        tmp = comp_matrix @ tf.expand_dims(
                (parameters_to_vector(weights) - w_c),
                axis = -1
            )
        coord_path.append(tf.squeeze(tmp))

    return tf.stack(coord_path)


def plot_training_path(
        coords: PCACoordinates, 
        training_path: list[tf.Tensor], 
        ax: plt.Axes = None, 
        end = None, 
        **kwargs
    ) -> plt.Axes:
    path = weights_to_coordinates(coords, training_path)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    colors = range(path.shape[0])
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0, end)
    ax.scatter(
        path[:, 0], path[:, 1], s=4, c=colors, cmap="cividis", norm=norm,
    )
    return ax
