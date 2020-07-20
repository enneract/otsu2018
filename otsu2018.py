import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition

from colour import (SpectralDistribution, STANDARD_OBSERVER_CMFS,
                    ILLUMINANT_SDS, sd_to_XYZ, XYZ_to_xy)
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


class Colours:
    """
    Represents multiple colours: their reflectances, XYZ tristimulus values
    and xy coordinates. The cmfs and the illuminant are taken from the parent
    Clustering.

    This class also supports partitioning, or creating two smaller instances
    of Colours, split along a horizontal or a vertical axis on the xy plane.
    """

    def __init__(self, clustering, reflectances):
        """
        Parameters
        ==========
        clustering : Clustering
            The parent clustering. This determines what cmfs and illuminant
            are used in colourimetric calculations.
        reflectances : ndarray (n,m)
            Reflectances of the ``n`` colours to be stored in this class.
            The shape must match ``Clustering.shape`` with ``m`` points for
            each colour.
        """

        self.reflectances = reflectances
        self.XYZ = np.empty((reflectances.shape[0], 3))
        self.xy = np.empty((reflectances.shape[0], 2))

        for i in range(len(self)):
            sd = SpectralDistribution(reflectances[i, :], clustering.wl)
            XYZ = sd_to_XYZ(sd, illuminant=clustering.illuminant) / 100
            self.XYZ[i, :] = XYZ
            self.xy[i, :] = XYZ_to_xy(XYZ)

    def __len__(self):
        """
        Counts the number of colours in this object.
        """
        return self.reflectances.shape[0]

    def partition(self, x_or_y, axis):
        """
        Parameters
        ==========
        x_or_y : int
            Whether to split according to X or Y coordinates.
        axis : float
            The coordinate that defines where the split happens.

        Returns
        =======
        lesser : Colours
            The left or lower part.
        greater : Colours
            The right or upper part.
        """
        mask = self.xy[:, x_or_y] <= axis

        lesser = object.__new__(Colours)
        greater = object.__new__(Colours)

        lesser.reflectances = self.reflectances[mask, :]
        greater.reflectances = self.reflectances[np.logical_not(mask), :]

        lesser.XYZ = self.XYZ[mask, :]
        greater.XYZ = self.XYZ[np.logical_not(mask), :]

        lesser.xy = self.xy[mask, :]
        greater.xy = self.xy[np.logical_not(mask), :]

        return lesser, greater


class ClusteringError(Exception):
    """
    Exception used for various errors originating from code in this file.
    """
    pass


class Node:
    """
    Represents a node in the clustering tree.
    """

    _counter = 1

    def __init__(self, clustering, colours):
        """
        Parameters
        ==========
        clustering : Clustering
            The parent clustering. This determines what cmfs and illuminant
            are used in colourimetric calculations.
        colours : Colours
            The colours that belong in this node.
        """

        self.clustering = clustering
        self.colours = colours
        self.children = None

        self._cached_reconstruction_error = None
        self.PCA_done = False

        # This is just for __str__ and plots
        self.number = Node._counter
        Node._counter += 1

    def __str__(self):
        return 'Node #%d' % (self.number)

    @property
    def leaf(self):
        """
        Is this node a leaf? Tree leaves don't have any children and store
        instances of ``Colours``.
        """

        return self.children is None

    def split(self, children, split_x_or_y, split_i):
        """
        Turns a leaf into a node with the given children.

        Parameters
        ==========
        children : tuple
            Two instances of ``Node`` in a tuple.
        split_x_or_y : int
            Split's ``x_or_y``.
        split_axis : float
            Split's ``axis``.
        """

        if not self.leaf:
            raise RuntimeError(
                'Node.split called for a node that is not a leaf')

        self.children = children
        self.split_x_or_y = split_x_or_y
        self.split_axis = self.colours.xy[split_i, split_x_or_y]
        self.colours = None
        self._cached_reconstruction_error = None

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

        pca = sklearn.decomposition.PCA(3)
        pca.fit(self.colours.reflectances)
        self.basis_functions = pca.components_
        self.mean = pca.mean_

        # TODO: better names
        M = np.empty((3, 3))
        for i in range(3):
            R = self.basis_functions[i, :]
            M[:, i] = self.clustering.fast_sd_to_XYZ(R)

        self.M_inverse = np.linalg.inv(M)
        self.XYZ_mu = self.clustering.fast_sd_to_XYZ(self.mean)

        self.PCA_done = True

    def reconstruct(self, XYZ):
        """
        Reconstructs a reflectance using data stored in this node.

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

        weights = np.dot(self.M_inverse, XYZ - self.XYZ_mu)
        reflectance = np.dot(weights, self.basis_functions) + self.mean
        return SpectralDistribution(reflectance, self.clustering.wl)

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

        if not self.PCA_done:  # FIXME
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

    def partition(self, x_or_y, i):
        """
        Splits this node into two and returns them. This operation does not
        affect the node it's used on. ``Node.split`` has to be called (with
        data returned from this method) to actually alter the tree.

        Parameters
        ==========
        x_or_y : int
            Whether to split according to X or Y coordinates.
        i : int
            The index of the colour whose coordinates determine where the
            split happens. Cannot be ``len(Node.colours)`` or greater.

        Returns
        =======
        lesser : Node
            The left or lower part.
        greater : Node
            The right or upper part.
        """

        axis = self.colours.xy[i, x_or_y]
        partition = self.colours.partition(x_or_y, axis)

        if len(partition[0]) <= 5 or len(partition[1]) <= 5:
            raise ClusteringError('partition created parts that are too small')

        lesser = Node(self.clustering, partition[0])
        greater = Node(self.clustering, partition[1])
        return lesser, greater

    def split_quality(self, x_or_y, i):
        """

        Parameters
        ==========
        x_or_y : int
            Whether to split according to X or Y coordinates.
        i : int
            Index of the colour whose coordinates determine where the
            split happens. Cannot be ``len(Node.colours)`` or greater.

        Returns
        =======
        error : float
            Sum of reconstruction errors of the two nodes created from
            splitting.
        lesser, greater : tuple
            Subnodes created from splitting.
        """
        lesser, greater = self.partition(x_or_y, i)

        lesser.PCA()
        greater.PCA()

        error = lesser.reconstruction_error() + greater.reconstruction_error()
        return error, (lesser, greater)

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


class Clustering:
    """
    Represents the process of clustering and optimisation. Instances store
    shared data such as cmfs and the illuminant.

    Operations involving the entire tree, such as optimisation and
    reconstruction, are implemented here.
    """

    def __init__(
            self,
            sds,
            shape,
            cmfs=STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer'],
            illuminant=ILLUMINANT_SDS['D65']):
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

        self.sds = sds
        self.shape = shape
        self.wl = shape.range()
        self.dw = self.wl[1] - self.wl[0]
        self.cmfs = cmfs.copy().align(shape)
        self.illuminant = illuminant.copy().align(shape)
        self.xy_w = XYZ_to_xy(sd_to_XYZ(illuminant, cmfs=cmfs))

        # The normalising constant used in sd_to_XYZ.
        self.k = 1 / (np.sum(self.cmfs.values[:, 1]
                      * self.illuminant.values) * self.dw)

        colours = Colours(self, sds)
        self.root = Node(self, colours)

    def fast_sd_to_XYZ(self, R):
        """
        Compute the XYZ tristimulus values of a given reflectance. Faster for
        humans, by using cmfs and the illuminant stored in the ''Clustering'',
        thus avoiding unnecessary repetition. Faster for computers, by using
        a very simple and direct method.

        Parameters
        ----------
        R : ndarray
            Reflectance with shape matching the one used to construct this
            ``Clustering``.

        Returns
        -------
        ndarray (3,)
            XYZ tristimulus values, normalised to 1.
        """

        E = self.illuminant.values * R
        return self.k * np.dot(E, self.cmfs.values) * self.dw

    def reconstruct(self, XYZ):
        """
        Finds the appropriate node and reconstructs the reflectance for the
        given XYZ tristimulus values.

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

        def search(node):
            if node.leaf:
                return node

            if xy[node.split_x_or_y] >= node.split_axis:
                return search(node.children[0])
            else:
                return search(node.children[1])

        node = search(self.root)
        return node.reconstruct(XYZ)

    def find_best_split(self):
        """
        Check every possible split in the entire tree to find the one that will
        reduce the error the most.

        Returns
        -------
        best_split : (Node, int, int)
            Tuple representing the best split found. It contains the ``Node``
            that should be split, the split direction (``x_or_y``) and
            the colour index (``i``).
        best_partition : (Node, Node)
            Subnodes to be used as children for the leaf.

        Raises
        ------
        ClusteringError
            If the tree has already been split too finely, further splits will
            not be possible and this exception will be raised.
        """

        best_new_error = None
        total_error = self.root.total_reconstruction_error()

        for leaf in self.root.leaves:
            total_error_minus_leaf = total_error - leaf.reconstruction_error()

            for x_or_y in [0, 1]:
                for i in range(len(leaf.colours)):
                    try:
                        split_error, partition = leaf.split_quality(x_or_y, i)
                    except ClusteringError:
                        continue

                    new_error = total_error_minus_leaf + split_error

                    if best_new_error is None or new_error < best_new_error:
                        best_new_error = new_error
                        best_split = (leaf, x_or_y, i)
                        best_partition = partition

                    print('%10s  %s %4d  %g'
                          % (leaf, ['x', 'y'][x_or_y], i, new_error))

        if best_new_error is None:
            raise ClusteringError('no more splits were possible')

        return best_split, best_partition

    def do_best_splits(self, repeats):
        """
        Find the best split and perform it, and repeat the operation the
        specified amount of times.

        Parameters
        ----------
        repeats : int
            Number of splits.
        """
        for repeat in range(repeats):
            try:
                (leaf, x_or_y, i), partition = self.find_best_split()
            except ClusteringError:
                print('WARNING: only %d splits were possible' % repeat)
                break

            print('==== Splitting %s, x_or_y=%d, i=%d ===='
                  % (leaf, x_or_y, i))
            leaf.split(partition, x_or_y, i)

    def write_python_dataset(self, path):
        """
        Write the clustering into a Python dataset compatible with Colour's
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
            for i, leaf in enumerate(self.root.leaves):
                leaf._i = i  # For use when writing the selection function
                for line in (repr(leaf.basis_functions) + ',').splitlines():
                    fd.write('    %s\n' % line)
            fd.write(']\n\n\n')

            # Means

            fd.write('OTSU_2018_MEANS = [\n')
            for leaf in self.root.leaves:
                for line in (repr(leaf.mean) + ',').splitlines():
                    fd.write('    %s\n' % line)
            fd.write(']\n\n\n')

            # Cluster selection function

            fd.write('def select_cluster_Otsu2018(xy):\n')
            fd.write('    x, y = xy\n\n')

            def write_if(node, indent):
                if node.leaf:
                    fd.write('    ' * indent)
                    fd.write('return %d  # %s\n' % (node._i, node))
                    return

                fd.write('    ' * indent)
                fd.write('if %s <= %s:\n' % (['x', 'y'][node.split_x_or_y],
                                             repr(node.split_axis)))
                write_if(node.children[0], indent + 1)

                fd.write('    ' * indent)
                fd.write('else:\n')
                write_if(node.children[1], indent + 1)

            write_if(self.root, 1)
