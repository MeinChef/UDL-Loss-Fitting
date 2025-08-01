from imports import tensorflow as tf
from imports import keras
from imports import numpy as np
from imports import skd
from scipy.interpolate import RegularGridInterpolator
from imports import plt
from imports import tqdm
from imports import plt3d #noqa


class PCACoordinates(object):
    def __init__(
            self, 
            training_path: list[tf.Tensor]
        ) -> None:

        """
            Class to handle the PCA coordinates of the training path. It computes the first two principal components
            of the training path and provides a callable to get the coordinates in the PCA space.
        """
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
            coords: PCACoordinates
        ) -> None:

        """
            Function to compile the loss surface. Since the surface we want to look at, is in high dimensions,
            it needs to be projected into a 2D space. This is done by using the first two principal components of the training path.
            The loss surface is then computed by evaluating the loss on a grid of points in this PCA space.

            :param range: Range of the grid in the PCA space.
            :type range: float
            :param points: Number of points in the grid.
            :type points: int
            :param coords: Coordinates object containing the PCA components.
            :type coords: PCACoordinates
            :return: None
            :rtype: None
        """

        a_grid = tf.linspace(-1.0, 1.0, num = points) ** 3 * range
        b_grid = tf.linspace(-1.0, 1.0, num = points) ** 3 * range
        loss_grid = np.empty([len(a_grid), len(b_grid)])

        print("Compiling loss surface...")
        for i, a in enumerate(tqdm.tqdm(a_grid)):
            for j, b in enumerate(b_grid):
                self.model_.set_weights(coords(a, b))

                loss_grid[j, i] = self.model_.test_on_batch(
                    self.inputs_, 
                    self.outputs_, 
                    return_dict=True
                )["loss"]

        print("Done!")
        self.model_.set_weights(coords.origin_)
        self.a_grid_ = a_grid
        self.b_grid_ = b_grid
        self.loss_grid_ = loss_grid

    def plot_surfc_and_loss(
        self,
        coords: PCACoordinates,
        training_path: list[tf.Tensor | np.ndarray],
        ax: plt.Axes = None,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        
        """
            Function for plotting the loss surface and the optimization/training path in the PCA space.

            :param coords: Coordinates object containing the PCA components.
            :type coords: PCACoordinates
            :param training_path: List of model weights representing the training path.
            :type training_path: list[tf.Tensor | np.ndarray]
            :param ax: Matplotlib Axes object to plot on. If None, a new figure and axes will be created.
            :type ax: plt.Axes, optional
            :param kwargs: Additional keyword arguments for the plot.
            :return: Tuple of the figure and axes objects.
            :rtype: tuple[plt.Figure, plt.Axes]
        """

        # project training path onto grid
        path = weights_to_coordinates(coords, training_path)

        # create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(
                nrows = 1,
                ncols = 1,
                subplot_kw = {'projection': '3d'},
                **kwargs
            )
            ax.set_aspect("equal")
            ax.set_title("The Loss Surface")
        
        # plot the surface
        X, Y = np.meshgrid(
            self.a_grid_,
            self.b_grid_
        )
        ax.plot_surface(
            X,
            Y,
            np.clip(self.loss_grid_, None, self.loss_grid_.min() + 5),
            cmap = "magma",
            alpha = 0.6,
            linewidth = 0,
            antialiased = True,
        )

        # interpolate the loss values for the training path
        interpolator = RegularGridInterpolator(
            (self.a_grid_, self.b_grid_),
            self.loss_grid_.T
        )
        z_loss_grid = interpolator(path)
        
        ax.scatter(
            path[:,0], 
            path[:,1],
            z_loss_grid,
            c = range(len(z_loss_grid)), 
            cmap = "plasma", 
            s = 15,
            zorder = 1
        )

        # make it pretty
        ax.tick_params(
            axis = 'both',
            which = 'major',
            labelsize = 8
        )
        ax.set_xlabel("PC1", fontsize = 12)
        ax.set_ylabel("PC2", fontsize = 12)
        ax.set_zlabel("Loss", fontsize = 12)

        return fig, ax


def weights_to_coordinates(
        coords: PCACoordinates, 
        training_path: list[tf.Tensor]
    ) -> tf.Tensor:
    """
        Project the training path onto the first two principal components using the pseudoinverse.
        
        :param coords: Coordinates object containing the PCA components.
        :type coords: PCACoordinates
        :param training_path: List of model weights representing the training path.
        :type training_path: list[tf.Tensor]
        :return: Tensor of shape (num_weights, 2) containing the coordinates in the PCA space.
        :rtype: tf.Tensor
    """

    components = [coords.v0_, coords.v1_]
    comp_matrix = paramlist_to_matrix(components)

 
    comp_matrix = tf.linalg.pinv(comp_matrix)

    # get the origin
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

    # convert list of model weights into a matrix of shape (#params, #steps)
    weight_matrix = paramlist_to_matrix(training_path)

    # create components
    pca = skd.PCA(
        n_components = n_components,
        whiten = True
    )
    components = pca.fit_transform(weight_matrix)
    components = tf.convert_to_tensor(
        components,
        dtype = tf.float32
    )
    
    # reshape to fit network
    example = training_path[0]
    weight_list = shape_weight_matrix_like(components, example)
    
    return pca, weight_list


def parameters_to_vector(
        params: list[np.ndarray | tf.Tensor]
    ) -> tf.Tensor:
    """
        Convience function for turning a model into a 1D-Tensor containing all variables.

        :param params: List of model weights.
        :type params: list[np.ndarray | tf.Tensor]
        :return: 1D Tensor containing all model weights.
        :rtype: tf.Tensor
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
        :return: Tensor of shape (#params, #steps)
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

    # split the long weights vector up into the respective weights
    weight_vecs = tf.split(
        weight_matrix, 
        weight_matrix.shape[1], 
        axis = 1
    )

    sizes = [tf.size(v).numpy() for v in example]
    shapes = [v.shape for v in example]
    
    weight_list = []
    for net_weights in weight_vecs:
        # split the long weights vector up into the respective sizes of each layer
        vs = tf.split(net_weights, sizes)

        # and reshape the now correct length vectors into the shape of the weights
        vs = [tf.reshape(v, s) for v, s in zip(vs, shapes)]
        weight_list.append(vs)
    
    return weight_list


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