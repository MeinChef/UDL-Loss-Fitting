from imports import tensorflow as tf
from imports import keras
from imports import numpy as np
from imports import skd
from scipy.interpolate import RegularGridInterpolator
from imports import plt
from imports import colors
from imports import tqdm
from imports import plt3d
from imports import plticker

# paper for visualising loss surfaces
# https://doi.org/10.48550/arXiv.1712.09913
# article using said article as source
# https://tvsujal.medium.com/visualising-the-loss-landscape-3a7bfa1c6fdf
# another article
# https://mathformachines.com/posts/visualizing-the-loss-landscape/


# TODO: annotate class
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

        print("Compiling loss surface...")
        for i, a in enumerate(tqdm.tqdm(a_grid)):
            for j, b in enumerate(b_grid):
                self.model_.set_weights(coords(a, b))

                loss_grid[j, i] = self.model_.test_on_batch(
                    self.inputs_, 
                    self.outputs_, 
                    return_dict=True
                )["loss"]
                # loss_grid[j, i] = loss

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
        
        path = weights_to_coordinates(coords, training_path)

        # since especially with the von Mises loss, the values of this array might be negative
        # substracting the minimum value from this array preserves the relationship between the
        # values, while the minimum becomes 0. Adding a small value (0.001) just shifts the whole thing 
        # to >0 (important when applying log)
        # zs = self.loss_grid_ - self.loss_grid_.min() + 0.001
        zs = self.loss_grid_

        # min_loss = zs.min()
        # max_loss = zs.max()


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
            np.clip(zs, None, 5),
            cmap = "magma",
            alpha = 0.6,
            linewidth = 0,
            antialiased = True,
            # norm = colors.LogNorm(vmin = min_loss, vmax = max_loss),
        )

        # calculate the actual loss
        # z_loss = np.empty(shape = (len(path),), dtype = np.float32)

        # for i, pth in enumerate(tqdm.tqdm(path)):
        #     self.model_.set_weights(coords(*pth))
        #     z_loss[i] = self.model_.test_on_batch(
        #         self.inputs_, 
        #         self.outputs_, 
        #         return_dict = True
        #     )["loss"]

        # ax.scatter(
        #     path[:,0], 
        #     path[:,1],
        #     z_loss,
        #     c = range(len(z_loss)), 
        #     cmap = "cividis", 
        #     s = 15,
        #     zorder = 1
        # )

        interpolator = RegularGridInterpolator((self.a_grid_, self.b_grid_), self.loss_grid_.T)
        z_loss_grid = interpolator(path)
        # print("z_loss (actual):", z_loss)
        # print("z_loss_grid (from grid):", z_loss_grid)
        ax.scatter(
            path[:,0], 
            path[:,1],
            z_loss_grid,
            c = range(len(z_loss_grid)), 
            cmap = "plasma", 
            s = 15,
            zorder = 1
        )
        
        # interpolate the z-value bilinear:
        # loss_pol = np.empty((len(path),))
        # for i, pth in enumerate(path):
        #     loss_pol[i] = self.custom_interpolator(pth, zs)
        # 
        # ax.scatter(
        #     path[:,0], 
        #     path[:,1],
        #     loss_pol,
        #     c = range(len(loss_pol)), 
        #     cmap = "plasma", 
        #     s = 15,
        #     # norm = colors.LogNorm(vmin = min_loss, vmax = max_loss),
        #     zorder = 1,
        # )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("Loss")

        return fig, ax

    def plot_contour(
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
            fig, ax = plt.subplots(
                nrows = 1,
                ncols = 1,
                # subplot_kw = {'projection': '3d'},
                **kwargs
            )
            ax.set_aspect("equal")
            ax.set_title("The Loss Surface")
        
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

    def plot_surface(
        self,
        levels: int = 20, 
        ax: plt.Axes = None, 
        show_contours = False, 
        **kwargs
    ) -> plt.Axes:
        
        xs = self.a_grid_
        ys = self.b_grid_
        zs = self.loss_grid_
        zs = zs.T

        # since especially with the von Mises loss, the values of this array might be negative
        # substracting the minimum value from this array preserves the relationship between the
        # values, while the minimum becomes 0. Adding a small value (0.001) just shifts the whole thing 
        # to >0 (important when applying log)
        # zs = zs - zs.min() + 0.001

        min_loss = zs.min()
        max_loss = zs.max()

        if ax is None:
            fig2, ax2 = plt.subplots(
                nrows = 1,
                ncols = 1,
                subplot_kw = {'projection': '3d'},
                **kwargs
            )
            ax2.set_title("The loss surface, but in 3D")
        
        X, Y = np.meshgrid(xs, ys)
        ax2.plot_surface(
            X,
            Y,
            np.clip(zs, None, 5),
            cmap = "magma",
            alpha = 0.75,
            linewidth = 0,
            antialiased = False,
            norm = colors.LogNorm(vmin = min_loss, vmax = max_loss),
        )

        # Optionally plot contour lines
        if show_contours:
            CS = ax2.contour(
                xs, 
                ys, 
                zs,
                levels = levels,
                c = "black",
                linestyles = "dotted",
                offset = 0.0,
                # norm = colors.LogNorm(vmin=zs.min(), vmax=zs.max()),
            )
            ax2.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")

        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("Loss")
        return ax2
    
    def plot_surface_and_path(
        self,
        coords: PCACoordinates,
        training_path: list[tf.Tensor],
        ax: plt.Axes = None,
        show_contours: bool = True,
        **kwargs
    ):
        path = weights_to_coordinates(coords, training_path).numpy()
        xs = self.a_grid_.numpy()
        ys = self.b_grid_.numpy()
        zs = self.loss_grid_

        # Transpose zs so that axes match meshgrid and interpolation
        zs = zs

        # Use the same zs for both interpolation and plotting
        interpolator = RegularGridInterpolator((xs, ys), zs)
        path_loss = interpolator(path)
        breakpoint()


        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, **kwargs)

        X, Y = np.meshgrid(xs, ys)
        surf = ax.plot_surface(
            X, Y, zs,
            cmap="magma",
            alpha=0.75,
            linewidth=0,
            antialiased=False,
            norm=colors.LogNorm(vmin=zs.min(), vmax=zs.max()),
        )

        if show_contours:
            levels = np.exp(np.linspace(np.log(zs.min()), np.log(zs.max()), 20))
            ax.contour(
                X, Y, zs,
                levels=levels,
                cmap="magma",
                linestyles="solid",
                offset=0.0,
                norm=colors.LogNorm(vmin=zs.min(), vmax=zs.max()),
            )

        # Plot the training path exactly on the surface
        # ax.plot(path[:, 0], path[:, 1], path_loss, color="black", linewidth=2, label="Training Path")
        ax.scatter(path[:, 0], path[:, 1], path_loss, c=range(len(path_loss)), cmap="cividis", s=15)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("Loss")
        # ax.legend()
        return ax
    
    def custom_interpolator(
        self,
        point: np.ndarray,
        zs: np.ndarray = None
    ) -> np.ndarray:
        """
        :param point: should be a 2-Element Array, with x, y respectively
        :return: 2-element array
        """

        x, y = point

        # get the closest point in the grid
        xidx = np.searchsorted(self.a_grid_, x, side = "left")
        yidx = np.searchsorted(self.b_grid_, y, side = "left")
        
        # get the x and y values from the grid
        x0, x1 = self.a_grid_[xidx], self.a_grid_[xidx + 1]
        y0, y1 = self.b_grid_[yidx], self.b_grid_[yidx + 1]

        if zs is None:
            zs = self.loss_grid_
        zs = zs.T

        # get the z values from the grid
        z00 = zs[xidx,   yidx]
        z10 = zs[xidx+1, yidx]
        z01 = zs[xidx,   yidx+1]
        z11 = zs[xidx+1, yidx+1]

        # calculate the weights for each diagonal part
        wx1 = (x - x0) / (x1 - x0)
        wx0 = 1 - wx1
        wy1 = (y - y0) / (y1 - y0)
        wy0 = 1 - wy1

        return (z00 * wx0 * wy0 +
                z10 * wx1 * wy0 +
                z01 * wx0 * wy1 +
                z11 * wx1 * wy1)

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

def plot_training_path_3d(
        coords: PCACoordinates, 
        training_path: list[tf.Tensor], 
        loss: list[float],
        ax = None, 
        end = None, 
        **kwargs
    ):
    path = weights_to_coordinates(coords, training_path)
    zs = loss  # Use step index or actual loss if available
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw = {"projection": "3d"},
            **kwargs
        )
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0, end)
    ax.scatter(
        path[:, 0], 
        path[:, 1], 
        zs, 
        s = 4,
        c = range(path.shape[0]),
        cmap = "cividis",
        norm = norm
    )
    return ax

def plot_training_path_on_surface(
        loss_surface: LossSurface,
        coords: PCACoordinates,
        training_path: list[tf.Tensor],
        ax: plt.Axes = None,
        **kwargs
    ):
    """
    Plots the training path as a 3D line on the loss surface, with optional height lines.
    """
    # Project training path to PCA coordinates
    path = weights_to_coordinates(coords, training_path).numpy()
    xs = loss_surface.a_grid_.numpy()
    ys = loss_surface.b_grid_.numpy()
    zs = loss_surface.loss_grid_

    # Transpose zs so that axes match meshgrid and interpolation
    zs = zs.T

    # Interpolate loss values at training path coordinates
    interpolator = RegularGridInterpolator((xs, ys), zs)
    path_loss = interpolator(path)

    # Create 3D axis if needed
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw = {"projection": "3d"},
            **kwargs
        )
    ax.scatter(
        path[:, 0], 
        path[:, 1], 
        path_loss, 
        c = range(len(path_loss)), 
        cmap = "cividis", 
        s = 15
    )

    return ax


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


