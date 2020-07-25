import numpy as np
import matplotlib.pyplot as plt
import time

from colour import (SpectralDistribution, STANDARD_OBSERVER_CMFS,
                    sd_ones, sd_to_XYZ, XYZ_to_xy)
from colour.plotting import plot_chromaticity_diagram_CIE1931


def load_Otsu2018_spectra(path, every_nth=1):
    """
    Loads a set of measured reflectances from Otsu et al.'s csv file.

    TODO: This function can't determine the spectral shape.

    Parameters
    ----------
    path : str
        File path.
    every_nth : int
        Load only every n-th spectrum. The original files are huge, so this can
        be useful for testing.
    """
    data = np.genfromtxt(path, delimiter=',', skip_header=1)

    # The first column is the id and is redundant
    data = data[:, 1:]

    spectra = []
    for i in range(data.shape[0]):
        if i % every_nth != 0:
            continue

        values = data[i, :]
        spectra.append(values)

    return np.array(spectra)


class PartitionAxis:
    """
    Represents a horizontal or vertical line, partitioning the 2D space in
    two half-planes.

    Attributes
    ----------
    origin : float
        The x coordinate of a vertical line or the y coordinate of a horizontal
        line.
    direction : int
        '0' if vertical, '1' if horizontal.
    """

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def __str__(self):
        return '%s=%s' % ('yx'[self.direction], repr(self.origin))


class Colours:
    """
    Represents multiple colours: their reflectances, XYZ tristimulus values
    and xy coordinates. The cmfs and the illuminant are taken from the parent
    tree.

    This class also supports partitioning, or creating two smaller instances
    of Colours, split along a horizontal or a vertical axis on the xy plane.
    """

    def __init__(self, tree, reflectances):
        """
        Parameters
        ==========
        tree : tree
            The parent tree. This determines what cmfs and illuminant
            are used in colourimetric calculations.
        reflectances : ndarray (n,m)
            Reflectances of the ``n`` colours to be stored in this class.
            The shape must match ``tree.shape`` with ``m`` points for
            each colour.
        """

        self.reflectances = reflectances
        self.XYZ = np.empty((reflectances.shape[0], 3))
        self.xy = np.empty((reflectances.shape[0], 2))

        for i in range(len(self)):
            sd = SpectralDistribution(reflectances[i, :], tree.wl)
            XYZ = sd_to_XYZ(sd, illuminant=tree.illuminant) / 100
            self.XYZ[i, :] = XYZ
            self.xy[i, :] = XYZ_to_xy(XYZ)

    def __len__(self):
        """
        Counts the number of colours in this object.
        """
        return self.reflectances.shape[0]

    def partition(self, axis):
        """
        Parameters
        ==========
        axis : PartitionAxis
            Defines the partition axis.

        Returns
        =======
        lesser : Colours
            The left or lower part.
        greater : Colours
            The right or upper part.
        """
        mask = self.xy[:, axis.direction] <= axis.origin

        lesser = object.__new__(Colours)
        greater = object.__new__(Colours)

        lesser.reflectances = self.reflectances[mask, :]
        greater.reflectances = self.reflectances[np.logical_not(mask), :]

        lesser.XYZ = self.XYZ[mask, :]
        greater.XYZ = self.XYZ[np.logical_not(mask), :]

        lesser.xy = self.xy[mask, :]
        greater.xy = self.xy[np.logical_not(mask), :]

        return lesser, greater


class treeError(Exception):
    """
    Exception used for various errors originating from code in this file.
    """
    pass


class Node:
    """
    Represents a node in the tree tree.
    """

    _counter = 1

    def __init__(self, tree, colours):
        """
        Parameters
        ==========
        tree : tree
            The parent tree. This determines what cmfs and illuminant
            are used in colourimetric calculations.
        colours : Colours
            The colours that belong in this node.
        """

        self.tree = tree
        self.colours = colours
        self.children = None

        self._cached_reconstruction_error = None
        self.PCA_done = False
        self.best_partition = None

        # This is just for __str__ and plots
        self.number = Node._counter
        Node._counter += 1

    def __str__(self):
        return 'Node #%d (%d)' % (self.number, len(self.colours))

    @property
    def leaf(self):
        """
        Is this node a leaf? Tree leaves don't have any children and store
        instances of ``Colours``.
        """

        return self.children is None

    def split(self, children, partition_axis):
        """
        Turns a leaf into a node with the given children.

        Parameters
        ==========
        children : tuple
            Two instances of ``Node`` in a tuple.
        partition_axis : PartitionAxis
            Defines the partition axis.
        """

        self.children = children
        self.partition_axis = partition_axis
        self.colours = None
        self._cached_reconstruction_error = None
        self.best_partition = None

    def _leaves_generator(self):
        if self.leaf:
            yield self
        else:
            for child in self.children:
                yield from child.leaves

    @property
    def leaves(self):
        """
        Returns a generator of all leaves connected to this node.
        """
        return self._leaves_generator()

    #
    # PCA and reconstruction
    #

    def PCA(self):
        """
        Performs the principal component analysis on colours in this node.
        """

        if not self.leaf:
            raise RuntimeError('Node.PCA called for a node that is not a leaf')

        self.mean = np.mean(self.colours.reflectances, axis=0)
        data_matrix = self.colours.reflectances - self.mean
        covariance_matrix = np.dot(data_matrix.T, data_matrix)
        _eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        self.basis_functions = eigenvectors[:, -3:].T

        # TODO: better names
        M = np.empty((3, 3))
        for i in range(3):
            R = self.basis_functions[i, :]
            M[:, i] = self.tree.fast_sd_to_XYZ(R)

        self.M_inverse = np.linalg.inv(M)
        self.XYZ_mu = self.tree.fast_sd_to_XYZ(self.mean)

        self.PCA_done = True

    def _reconstruct_xy(self, XYZ, xy):
        if not self.leaf:
            if xy[self.partition_axis.direction] <= self.partition_axis.origin:
                return self.children[0]._reconstruct_xy(XYZ, xy)
            else:
                return self.children[1]._reconstruct_xy(XYZ, xy)

        weights = np.dot(self.M_inverse, XYZ - self.XYZ_mu)
        reflectance = np.dot(weights, self.basis_functions) + self.mean
        reflectance = np.clip(reflectance, 0, 1)
        return SpectralDistribution(reflectance, self.tree.wl)

    def reconstruct(self, XYZ):
        """
        Reconstructs the reflectance for the given *XYZ* tristimulus values.
        If this is a leaf, data from this node will be used. Otherwise the
        code will look for the appropriate subnode.

        Parameters
        ==========
        XYZ : ndarray, (3,)
            *CIE XYZ* tristimulus values to recover the spectral distribution
            from.

        Returns
        -------
        SpectralDistribution
            Recovered spectral distribution.
        """

        xy = XYZ_to_xy(XYZ)
        return self._reconstruct_xy(XYZ, xy)

    #
    # Optimisation
    #

    def reconstruction_error(self):
        """
        For every colour in this node, its spectrum is reconstructed (using
        PCA data in this node) and compared with its true, measured spectrum.
        The errors are then summed up and returned.

        Returns
        =======
        error : float
            The sum reconstruction errors for this node.
        """

        if self._cached_reconstruction_error:
            return self._cached_reconstruction_error

        if not self.PCA_done:
            self.PCA()

        error = 0
        for i in range(len(self.colours)):
            sd = self.colours.reflectances[i, :]
            XYZ = self.colours.XYZ[i, :]
            recovered_sd = self.reconstruct(XYZ)
            error += np.sum((sd - recovered_sd.values) ** 2)

        self._cached_reconstruction_error = error
        return error

    def total_reconstruction_error(self):
        """
        Computes the reconstruction error for an entire subtree, starting at
        this node.

        Returns
        =======
        error : float
            The total reconstruction error of the subtree.
        """

        if self.leaf:
            return self.reconstruction_error()
        else:
            return sum([child.total_reconstruction_error()
                        for child in self.children])

    def partition_error(self, axis):
        """
        Compute the sum of reconstruction errors of the two nodes created by
        a given partition of this node.

        Parameters
        ==========
        axis : PartitionAxis
            Defines the partition axis.

        Returns
        =======
        error : float
            Sum of reconstruction errors of the two nodes created from
            splitting.
        lesser, greater : tuple
            Subnodes created from splitting.
        """
        partition = self.colours.partition(axis)

        if (len(partition[0]) < self.tree.min_cluster_size
                or len(partition[1]) < self.tree.min_cluster_size):
            raise treeError(
                'partition created parts smaller than min_cluster_size')

        lesser = Node(self.tree, partition[0])
        lesser.PCA()

        greater = Node(self.tree, partition[1])
        greater.PCA()

        error = lesser.reconstruction_error() + greater.reconstruction_error()
        return error, (lesser, greater)

    def find_best_partition(self):
        """
        Finds the best partition of this node. See
        ``tree.find_best_partition``.
        """

        if self.best_partition is not None:
            return self.best_partition

        error = self.reconstruction_error()
        best_error = None

        bar = None
        if self.tree._progress_bar:
            bar = self.tree._progress_bar(total=2 * len(self.colours),
                                          leave=False)

        for direction in [0, 1]:
            for i in range(len(self.colours)):
                if bar:
                    bar.update()

                origin = self.colours.xy[i, direction]
                axis = PartitionAxis(origin, direction)

                try:
                    new_error, partition = self.partition_error(axis)
                except treeError:
                    continue

                if new_error >= error:
                    continue

                if best_error is None or new_error < best_error:
                    self.best_partition = (new_error, axis, partition)

        if bar:
            bar.close()

        if self.best_partition is None:
            raise treeError('no partitions are possible')

        return self.best_partition

    #
    # Plotting
    #

    def _plot_colours(self, number):
        if not self.leaf:
            for child in self.children:
                child._plot_colours(number)
            return

        symbols = ['+', '^', '*', '>', 'o', 'v', 'x', '<']
        plt.plot(*self.colours.xy.T,
                 "k" + symbols[number[0] % len(symbols)],
                 label=str(self))
        number[0] += 1

    def visualise(self):
        """
        Plots the subtree on a xy chromaticity diagram. Does not call
        ``plt.show``.
        """
        plot_chromaticity_diagram_CIE1931(standalone=False)
        self._plot_colours([0])
        plt.legend()

    def visualise_pca(self):
        """
        Makes a plot showing the principal components of this node and how
        well they reconstruct the source data.
        """

        plt.subplot(2, 1, 1)
        plt.title(str(self) + ': principal components')
        for i in range(3):
            plt.plot(self.wl, self.basis_functions[i, :], label='PC%d' % i)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title(str(self) + ': data')
        for i in range(3):
            plt.plot(self.wl, self.colours.reflectances[i, :], 'C%d:' % i)

            XYZ = self.colours.XYZ[i, :]
            recon = self.reconstruct(XYZ)
            plt.plot(self.wl, recon.values, 'C%d-' % i)
            recon = self.reconstruct(XYZ)
            plt.plot(self.wl, recon.values, 'C%d--' % i)


class Tree(Node):
    """
    This is an extension of ``Node``. It's meant to represent the root of the
    tree and contains information shared with all the nodes, such as cmfs
    and the illuminant (if any is used).

    Operations involving the entire tree, such as optimisation and
    reconstruction, are also implemented here.
    """

    def __init__(
            self,
            sds,
            shape,
            cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer'],
            illuminant=sd_ones()):
        """
        Parameters
        ----------
        sds : ndarray (n,m)
            Reflectances of the ``n`` reference colours to be used for
            optimisation.
        shape : SpectralShape
            Spectral shape of ``sds``.
        cmfs : XYZ_ColourMatchingFunctions, optional
            Standard observer colour matching functions.
        illuminant : SpectralDistribution, optional
            Illuminant spectral distribution.
        """

        self.shape = shape
        self.wl = shape.range()
        self.dw = self.wl[1] - self.wl[0]
        self.cmfs = cmfs.copy().align(shape)
        self.illuminant = illuminant.copy().align(shape)
        self.xy_w = XYZ_to_xy(sd_to_XYZ(illuminant, cmfs=cmfs))

        # The normalising constant used in sd_to_XYZ.
        self.k = 1 / (np.sum(self.cmfs.values[:, 1]
                      * self.illuminant.values) * self.dw)

        super().__init__(self, Colours(self, sds))

    def fast_sd_to_XYZ(self, R):
        """
        Compute the XYZ tristimulus values of a given reflectance. Faster for
        humans, by using cmfs and the illuminant stored in the ''tree'',
        thus avoiding unnecessary repetition. Faster for computers, by using
        a very simple and direct method.

        Parameters
        ----------
        R : ndarray
            Reflectance with shape matching the one used to construct this
            ``tree``.

        Returns
        -------
        ndarray (3,)
            XYZ tristimulus values, normalised to 1.
        """

        E = self.illuminant.values * R
        return self.k * np.dot(E, self.cmfs.values) * self.dw

    def optimise(self,
                 repeats=8,
                 min_cluster_size=None,
                 print_callback=print,
                 progress_bar=None):
        """
        Optimise the tree by repeatedly performing optimal partitions of the
        nodes, creating a tree that minimizes the total reconstruction
        error.

        Parameters
        ----------
        repeats : int, optional
            Maximum number of splits. If the dataset is too small, this number
            might not be reached. The default is to create 8 clusters, like in
            the original paper.
        min_cluster_size : int, optional
            Smallest acceptable cluster size. By default it's chosen
            automatically, based on the size of the dataset and desired number
            of clusters. Must be at least 3 or principal component analysis
            will not be possible.
        print_callback : function, optional
            Function to use for printing progress and diagnostic information.
        progress_bar : class, optional
            Class for creating progress bar objects. Must be compatible with
            tqdm.
        """

        t0 = time.time()

        def _print(text):
            if print_callback is None:
                return

            delta = time.time() - t0
            stamp = '%3d:%02d ' % (delta // 60, np.floor(delta % 60))
            for line in text.splitlines():
                print_callback(stamp, line)

        self._progress_bar = progress_bar

        if min_cluster_size is not None:
            self.min_cluster_size = min_cluster_size
        else:
            self.min_cluster_size = len(self.colours) / repeats // 2

        if self.min_cluster_size <= 3:
            self.min_cluster_size = 3

        initial_error = self.total_reconstruction_error()
        _print('Initial error is %g.' % initial_error)

        for repeat in range(repeats):
            _print('\n=== Iteration %d of %d ===' % (repeat + 1, repeats))

            best_total_error = None
            total_error = self.total_reconstruction_error()

            for i, leaf in enumerate(self.leaves):
                _print('Optimising %s...' % leaf)

                try:
                    error, axis, partition = leaf.find_best_partition()
                except treeError as e:
                    _print('Failed: %s' % e)
                    continue

                new_total_error = (total_error - leaf.reconstruction_error()
                                   + error)
                if (best_total_error is None
                        or new_total_error < best_total_error):
                    best_total_error = new_total_error
                    best_axis = axis
                    best_leaf = leaf
                    best_partition = partition

            if best_total_error is None:
                _print('\nNo further improvements are possible.\n'
                       'Terminating at iteration %d.\n' % repeat)
                break

            _print('\nSplit %s into %s and %s along %s.'
                   % (best_leaf, *best_partition, best_axis))
            _print('Error is reduced by %g and is now %g, '
                   '%.1f%% of the initial error.'
                   % (leaf.reconstruction_error()
                      - error, best_total_error, 100 * best_total_error
                      / initial_error))

            best_leaf.split(best_partition, best_axis)

        _print('Finished.')

    def write_python_dataset(self, path):
        """
        Write the tree into a Python dataset compatible with Colour's
        ``colour.recovery.otsu2018`` code.

        Parameters
        ----------
        path : string
            File path.
        """

        with open(path, 'w') as fd:
            fd.write('# Autogenerated, do not modify\n\n')

            fd.write('from numpy import array\n')
            fd.write('from colour import SpectralShape\n\n\n')

            fd.write('OTSU_2018_SPECTRAL_SHAPE = SpectralShape%s\n\n\n'
                     % self.shape)

            # Basis functions

            fd.write('OTSU_2018_BASIS_FUNCTIONS = [\n')
            for i, leaf in enumerate(self.leaves):
                for line in (repr(leaf.basis_functions) + ',').splitlines():
                    fd.write('    %s\n' % line)
            fd.write(']\n\n\n')

            # Means

            fd.write('OTSU_2018_MEANS = [\n')
            for leaf in self.leaves:
                for line in (repr(leaf.mean) + ',').splitlines():
                    fd.write('    %s\n' % line)
            fd.write(']\n\n\n')

            # Cluster selection function

            fd.write('def select_cluster_Otsu2018(xy):\n')
            fd.write('    x, y = xy\n\n')

            counter = 0

            def write_if(node, indent):
                nonlocal counter

                if node.leaf:
                    fd.write('    ' * indent)
                    fd.write('return %d  # %s\n' % (counter, node))
                    counter += 1
                    return

                fd.write('    ' * indent)
                fd.write('if %s <= %s:\n'
                         % ('xy'[node.partition_axis.direction],
                            repr(node.partition_axis.origin)))
                write_if(node.children[0], indent + 1)

                fd.write('    ' * indent)
                fd.write('else:\n')
                write_if(node.children[1], indent + 1)

            write_if(self, 1)
