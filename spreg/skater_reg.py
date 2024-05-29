"""Skater Regression classes."""

__author__ = "Luc Anselin anselin@uchicago.edu, Pedro Amaral pedroamaral@cedeplar.ufmg.br, Levi Wolf levi.john.wolf@bristol.ac.uk"

from scipy.sparse import csgraph as cg
from scipy.optimize import OptimizeWarning
from collections import namedtuple
from warnings import warn
from libpysal.weights import w_subset
from .utils import set_endog
from .twosls_regimes import TSLS_Regimes
import time
import numpy as np
import copy

try:
    from sklearn.metrics import euclidean_distances
except ImportError:
    from scipy.spatial.distance import pdist, cdist, squareform

    def euclidean_distances(X, Y=None):
        """
        fallback function to compute pairwise euclidean distances
        for a single input, or point-to-point euclidean distances
        for two inputs.
        """
        if Y is None:
            return squareform(pdist(X))
        else:
            return cdist(X, Y)


__all__ = ["Skater_reg"]

deletion = namedtuple("deletion", ("in_node", "out_node", "score"))


class Skater_reg(object):
    """
    Initialize the Skater_reg algorithm based on :cite:`Anselin2021`.
    The function can currently estimate OLS, from
    spreg or stats_models, and Spatial Lag models from spreg.
    Fit method performs estimation and returns a Skater_reg object.

    Parameters
    ----------
    dissimilarity : a callable distance metric.
                    Default: sklearn.metrics.pairwise.euclidean_distances
    affinity      : a callable affinity metric between 0,1.
                    Will be inverted to provide a
                    dissimilarity metric.
    reduction     : the reduction applied over all clusters
                    to provide the map score.
                    Default: numpy.sum
    center        : way to compute the center of each region in attribute space
                    Default: numpy.mean

    NOTE: Optimization occurs with respect to a *dissimilarity* metric, so the reduction should
              yield some kind of score where larger values are *less desirable* than smaller values.
              Typically, this means we use addition.


    Attributes
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates used for
                    calibration locations
    y             : array
                    n*1, dependent variable
    X             : array
                    n*k, independent variable, not including the constant
    bw            : scalar
                    bandwidth value consisting of either a distance or N
                    nearest neighbors; user specified or obtained using
                    Sel_BW
    family        : family object
                    underlying probability model; provides
                    distribution-specific calculations
    offset        : array
                    n*1, the offset variable at the ith location. For Poisson model
                    this term is often the size of the population at risk or
                    the expected size of the outcome in spatial epidemiology
                    Default is None where Ni becomes 1.0 for all locations
    sigma2_v1     : boolean
                    specify form of corrected denominator of sigma squared to use for
                    model diagnostics; Acceptable options are:
                    'True':       n-tr(S) (defualt)
                    'False':     n-2(tr(S)+tr(S'S))
    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'
    fixed         : boolean
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)
    constant      : boolean
                    True to include intercept (default) in model and False to exclude
                    intercept
    spherical     : boolean
                    True for shperical coordinates (long-lat),
                    False for projected coordinates (defalut).
    hat_matrix    : boolean
                    True to store full n by n hat matrix,
                    False to not store full hat matrix to minimize memory footprint (defalut).
    n             : integer
                    number of observations
    k             : integer
                    number of independent variables
    mean_y        : float
                    mean of y
    std_y         : float
                    standard deviation of y
    fit_params    : dict
                    parameters passed into fit method to define estimation
                    routine
    points        : array-like
                    n*2, collection of n sets of (x,y) coordinates used for
                    calibration locations instead of all observations;
                    defaults to None unles specified in predict method
    P             : array
                    n*k, independent variables used to make prediction;
                    exlcuding the constant; default to None unless specified
                    in predict method
    exog_scale    : scalar
                    estimated scale using sampled locations; defualt is None
                    unless specified in predict method
    exog_resid    : array-like
                    estimated residuals using sampled locations; defualt is None
                    unless specified in predict method
    Examples
    --------
    >>> import libpysal as ps
    >>> import numpy as np
    >>> import spreg
    >>> from spreg.skater_reg import Skater_reg
    >>> data = ps.io.open(ps.examples.get_path('columbus.dbf'))
    >>> y = np.array(data.by_col('HOVAL')).reshape((-1,1))
    >>> x_var = ['INC','CRIME']
    >>> x = np.array([db.by_col(name) for name in x_var]).T
    >>> w = ps.weights.Queen.from_shapefile(ps.examples.get_path("columbus.shp"))
    >>> x_std = (x - np.mean(x,axis=0)) / np.std(x,axis=0)
    >>> results = Skater_reg().fit(3, w, x_std, {'reg':spreg.OLS,'y':y,'x':x}, quorum=10, trace=False)
    >>> results.current_labels_
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
       0, 1, 0, 0, 2, 2, 0, 0, 1, 0, 2, 1, 2, 1, 2, 0, 1, 0, 0, 1, 2, 2,
       2, 1, 0, 2, 2], dtype=int32)
    """

    def __init__(
        self,
        dissimilarity=euclidean_distances,
        affinity=None,
        reduction=np.sum,
        center=np.mean,
    ):
        if affinity is not None:
            # invert the 0,1 affinity to
            # to an unbounded positive dissimilarity
            metric = lambda x: -np.log(affinity(x))
        else:
            metric = dissimilarity
        self.metric = metric
        self.reduction = reduction
        self.center = center

    def __repr__(self):
        return "Skater_reg_object(metric = {}, reduction = {}, center = {})".format(
            self.metric, self.reduction, self.center
        )

    def fit(
        self,
        n_clusters,
        W,
        data=None,
        data_reg=None,
        quorum=-np.inf,
        trace=True,
        islands="increase",
        verbose=False,
        model_family="spreg",
    ):
        """
        Method that fits a model with a particular estimation routine.

        Parameters
        ----------
        n_clusters   : int of clusters wanted
        W            : pysal W object expressing the neighbor relationships between observations.
                       Should be symmetric and binary, so Queen/Rook, DistanceBand, or a symmetrized KNN.
        data         : np.ndarray of (N,P) shape with N observations and P features
                       This is the data that is used to evaluate the similarity between each observation.
        data_reg     : list containing:
                       1- a callable regression method (ex. OLS or GM_Lag from spreg or OLS from statsmodels)
                       2- np.ndarray of (N,1) shape with N observations on the depedent variable for the regression
                       3- np.ndarray of (N,k) shape with N observations and k columns containing the explanatory variables (constant must not be included)
                       4- pysal W object to be used in the regression (optional)
        quorum       : int with minimum size of each region.
        trace        : bool denoting whether to store intermediate
                       labelings as the tree gets pruned
        islands      : string describing what to do with islands.
                       If "ignore", will discover `n_clusters` regions, treating islands as their own regions.
                       If "increase", will discover `n_clusters` regions, treating islands as separate from n_clusters.
        verbose      : bool/int describing how much output to provide to the user,
                       in terms of print statements or progressbars.
        model_family : string describing the fFamily of estimation method used for the regression.
                       Must be either 'spreg' (default) or 'statsmodels'

        Returns
        -------
                     : Skater_reg object.
        """
        if trace:
            self._trace = []
        if data is None:
            attribute_kernel = np.ones((W.n, W.n))
            data = np.ones((W.n, 1))
        else:
            attribute_kernel = self.metric(data)
        W.transform = "b"
        W = W.sparse
        start = time.time()

        super_verbose = verbose > 1
        start_W = time.time()
        dissim = W.multiply(attribute_kernel)
        dissim.eliminate_zeros()
        end_W = time.time() - start_W

        if super_verbose:
            print("Computing Affinity Kernel took {:.2f}s".format(end_W))

        tree_time = time.time()
        MSF = cg.minimum_spanning_tree(dissim)
        tree_time = time.time() - tree_time
        if super_verbose:
            print("Computing initial MST took {:.2f}s".format(tree_time))

        initial_component_time = time.time()
        current_n_subtrees, current_labels = cg.connected_components(
            MSF, directed=False
        )
        initial_component_time = time.time() - initial_component_time

        if super_verbose:
            print(
                "Computing connected components took {:.2f}s.".format(
                    initial_component_time
                )
            )

        if current_n_subtrees > 1:
            island_warnings = [
                "Increasing `n_clusters` from {} to {} in order to account for islands.".format(
                    n_clusters, n_clusters + current_n_subtrees
                ),
                "Counting islands towards the remaining {} clusters.".format(
                    n_clusters - (current_n_subtrees)
                ),
            ]
            ignoring_islands = int(islands.lower() == "ignore")
            chosen_warning = island_warnings[ignoring_islands]
            warn(
                "By default, the graph is disconnected! {}".format(chosen_warning),
                OptimizeWarning,
                stacklevel=2,
            )
            if not ignoring_islands:
                n_clusters += current_n_subtrees
            _, island_populations = np.unique(current_labels, return_counts=True)
            if (island_populations < quorum).any():
                raise ValueError(
                    "Islands must be larger than the quorum. If not, drop the small islands and solve for"
                    " clusters in the remaining field."
                )
        if trace:
            self._trace.append(([], deletion(np.nan, np.nan, np.inf)))
            if super_verbose:
                print(self._trace[-1])
        trees_scores = None
        prev_score = np.inf
        while current_n_subtrees < n_clusters:  # while we don't have enough regions
            (
                best_deletion,
                trees_scores,
                new_MSF,
                current_n_subtrees,
                current_labels,
            ) = self.find_cut(
                MSF,
                data,
                data_reg,
                current_n_subtrees,
                current_labels,
                quorum=quorum,
                trees_scores=trees_scores,
                labels=None,
                target_label=None,
                verbose=verbose,
                model_family=model_family,
            )

            if np.isfinite(best_deletion.score):  # if our search succeeds
                # accept the best move as *the* move
                if super_verbose:
                    print("cut made {}...".format(best_deletion))
                if best_deletion.score > prev_score:
                    raise ValueError(
                        ("The score increased with the number of clusters. "
                            "Please check your data.\nquorum: {}; n_clusters: {}"
                        ).format(quorum, n_clusters)
                    )
                prev_score = best_deletion.score
                MSF = new_MSF
            else:  # otherwise, it means the MSF admits no further cuts
                prev_n_subtrees, _ = cg.connected_components(MSF, directed=False)
                warn(
                    "MSF contains no valid moves after finding {} subtrees. "
                    "Decrease the size of your quorum to find the remaining {} subtrees.".format(
                        prev_n_subtrees, n_clusters - prev_n_subtrees
                    ),
                    OptimizeWarning,
                    stacklevel=2,
                )
            if trace:
                self._trace.append((current_labels, best_deletion))

        self.current_labels_ = current_labels
        self.minimum_spanning_forest_ = MSF
        self._elapsed_time = time.time() - start
        return self

    def score_spreg(
        self,
        data=None,
        data_reg=None,
        all_labels=None,
        quorum=-np.inf,
        current_labels=None,
        current_tree=None,
    ):
        """
        This yields a score for the data using methods from the spreg library, given the labels provided. If no labels are provided,
        and the object has been fit, then the labels discovered from the previous fit are used.

        If a quorum is not passed, it is assumed to be irrelevant.

        If a quorum is passed and the labels do not meet quorum, the score is inf.

        data        :   (N,P) array of data on which to compute the score of the regions expressed in labels
        data_reg    :   list containing:
                        1- a callable spreg regression method (ex. OLS or GM_Lag)
                        2- np.ndarray of (N,1) shape with N observations on the depedent variable for the regression
                        3- np.ndarray of (N,k) shape with N observations and k columns containing the explanatory variables (constant must not be included)
                        4- pysal W object to be used in the regression (optional)
        all_labels  :   (N,) flat vector of labels expressing the classification of each observation into a region considering the cut under evaluation.
        quorum      :   int expressing the minimum size of regions. Can be -inf if there is no lower bound.
                        Any region below quorum makes the score inf.
        current_labels: (N,) flat vector of labels expressing the classification of each observation into a region not considering the cut under evaluation.

        current_tree: integer indicating the tree's label currently being considered for division
        """

        labels, subtree_quorums = self._prep_score(
            all_labels, current_tree, current_labels
        )
        if (subtree_quorums < quorum).any():
            return np.inf, None
        set_labels = set(labels)
        if data_reg is not None:
            kargs = {
                k: v
                for k, v in data_reg.items()
                if k not in ["reg", "y", "x", "w", "x_nd"]
            }
            trees_scores = {}

            if data_reg["reg"].__name__ == "GM_Lag" or data_reg["reg"].__name__ == "BaseGM_Lag":
                try:
                    x = np.hstack((np.ones((data_reg["x"].shape[0], 1)), data_reg["x"]))
                    reg = TSLS_Regimes(
                        y=data_reg["y"],
                        x=x,
                        yend=data_reg["yend"],
                        q=data_reg["q"],
                        regimes=all_labels,)
                except:
                    x = _const_x(data_reg["x"])
                    reg = TSLS_Regimes(
                        y=data_reg["y"],
                        x=x,
                        yend=data_reg["yend"],
                        q=data_reg["q"],
                        regimes=all_labels,)
                score = np.dot(reg.u.T, reg.u)[0][0]
            else:

                for l in set_labels:
                    x = data_reg["x"][all_labels == l]
                    if np.linalg.matrix_rank(x) < x.shape[1]:
                        small_diag_indices = np.abs(np.diag(np.linalg.qr(x)[1])) < 1e-10
                        x = x[:, ~small_diag_indices]

                    if "w" not in data_reg:
                        try:
                            x = np.hstack((np.ones((x.shape[0], 1)), x))
                            reg = data_reg["reg"](
                                y=data_reg["y"][all_labels == l], x=x, **kargs
                            )
                        except np.linalg.LinAlgError:
                            x = _const_x(x)
                            reg = data_reg["reg"](
                                y=data_reg["y"][all_labels == l], x=x, **kargs
                            )
                    else:
                        l_arrays = np.array(all_labels)

                        regi_ids = list(np.where(l_arrays == l)[0])
                        w_ids = list(map(data_reg["w"].id_order.__getitem__, regi_ids))
                        w_regi_i = w_subset(data_reg["w"], w_ids, silence_warnings=True)
                        try:
                            x = np.hstack((np.ones((x.shape[0], 1)), x))
                            reg = data_reg["reg"](
                                y=data_reg["y"][all_labels == l], x=x, w=w_regi_i, **kargs
                            )
                        except np.linalg.LinAlgError:
                            x = _const_x(x)
                            reg = data_reg["reg"](
                                y=data_reg["y"][all_labels == l], x=x, w=w_regi_i, **kargs
                            )
                    trees_scores[l] = np.dot(reg.u.T, reg.u)[0][0]
                score = sum(trees_scores.values())
        else:
            part_scores, score, trees_scores = self._data_reg_none(
                data, all_labels, l, set_labels
            )

        return score, trees_scores

    def score_stats(
        self,
        data=None,
        data_reg=None,
        all_labels=None,
        quorum=-np.inf,
        current_labels=None,
        current_tree=None,
    ):
        """
        This yields a score for the data using methods from the stats_models library, given the labels provided. If no labels are provided,
        and the object has been fit, then the labels discovered from the previous fit are used.

        If a quorum is not passed, it is assumed to be irrelevant.

        If a quorum is passed and the labels do not meet quorum, the score is inf.

        data        :   (N,P) array of data on which to compute the score of the regions expressed in labels
        data_reg    :   list containing:
                        1- a callable statsmodels regression method (ex. OLS)
                        2- np.ndarray of (N,1) shape with N observations on the depedent variable for the regression
                        3- np.ndarray of (N,k) shape with N observations and k columns containing the explanatory variables (constant must not be included)
                        4- pysal W object to be used in the regression (optional)
        all_labels  :   (N,) flat vector of labels expressing the classification of each observation into a region considering the cut under evaluation.
        quorum      :   int expressing the minimum size of regions. Can be -inf if there is no lower bound.
                        Any region below quorum makes the score inf.
        current_labels: (N,) flat vector of labels expressing the classification of each observation into a region not considering the cut under evaluation.

        current_tree: integer indicating the tree label is currently being considered for division

        NOTE: Optimization occurs with respect to a *dissimilarity* metric, so the problem *minimizes*
              the map dissimilarity. So, lower scores are better.
        """
        labels, subtree_quorums = self._prep_score(
            all_labels, current_tree, current_labels
        )
        if (subtree_quorums < quorum).any():
            return np.inf, None
        set_labels = set(labels)
        if data_reg is not None:
            kargs = {
                k: v
                for k, v in data_reg.items()
                if k not in ["reg", "y", "x", "w", "x_nd"]
            }
            trees_scores = {}
            for l in set_labels:
                x = data_reg["x"][all_labels == l]
                if np.linalg.matrix_rank(x) < x.shape[1]:
                    small_diag_indices = np.abs(np.diag(np.linalg.qr(x)[1])) < 1e-10
                    x = x[:, ~small_diag_indices]

                try:
                    x = np.hstack((np.ones((x.shape[0], 1)), x))
                    reg = data_reg["reg"](
                        data_reg["y"][all_labels == l], x, **kargs
                    ).fit()
                except np.linalg.LinAlgError:
                    x = _const_x(x)
                    reg = data_reg["reg"](
                        data_reg["y"][all_labels == l], x, **kargs
                    ).fit()

                trees_scores[l] = np.sum(reg.resid ** 2)
            score = sum(trees_scores.values())
        else:
            part_scores, score, trees_scores = self._data_reg_none(
                data, all_labels, l, set_labels
            )
        return score, trees_scores

    def _prep_score(self, all_labels, current_tree, current_labels):
        if all_labels is None:
            try:
                labels = self.current_labels_
            except AttributeError:
                raise ValueError(
                    "Labels not provided and MSF_Prune object has not been fit to data yet."
                )
        if current_tree is not None:
            labels = all_labels[current_labels == current_tree]
        _, subtree_quorums = np.unique(labels, return_counts=True)
        return labels, subtree_quorums

    def _data_reg_none(self, data, all_labels, l, set_labels):
        assert data.shape[0] == len(
            all_labels
        ), "Length of label array ({}) does not match " "length of data ({})! ".format(
            all_labels.shape[0], data.shape[0]
        )
        part_scores = [
            self.reduction(
                self.metric(
                    X=data[all_labels == l],
                    Y=self.center(data[all_labels == l], axis=0).reshape(1, -1),
                )
            )
            for l in set_labels
        ]

        score = self.reduction(part_scores).item()
        trees_scores = {l: part_scores[i] for i, l in enumerate(set_labels)}
        return part_scores, score, trees_scores

    def _prep_lag(self, data_reg):
        # if the model is a spatial lag, add the lagged dependent variable to the model
        data_reg['yend'], data_reg['q'] = set_endog(data_reg["y"], data_reg["x"][:, 1:], data_reg["w"], yend=None,
            q=None, w_lags=1, lag_q=True)
        return data_reg

    def find_cut(
        self,
        MSF,
        data=None,
        data_reg=None,
        current_n_subtrees=None,
        current_labels=None,
        quorum=-np.inf,
        trees_scores=None,
        labels=None,
        target_label=None,
        make=False,
        verbose=False,
        model_family="spreg",
    ):
        """
        Find the best cut from the MSF.

        MSF: (N,N) scipy sparse matrix with zero elements removed.
             Represents the adjacency matrix for the minimum spanning forest.
             Constructed from sparse.csgraph.sparse_from_dense or using MSF.eliminate_zeros().
             You MUST remove zero entries for this to work, otherwise they are considered no-cost paths.
        data: (N,p) attribute matrix. If not provided, replaced with (N,1) vector of ones.
        data_reg: optional list containing:
                        1- a callable spreg or statsmodels regression method (ex. OLS or GM_Lag)
                        2- np.ndarray of (N,1) shape with N observations on the depedent variable for the regression
                        3- np.ndarray of (N,k) shape with N observations and k columns containing the explanatory variables (constant must not be included)
                        4- pysal W object to be used in the regression (optional)
        current_n_subtrees: integer indication the current number of subtrees.
        current_labels: (N,) flat vector of labels expressing the classification of each observation into a region not considering the cut under evaluation.
        quorum: int denoting the minimum number of elements in the region
        trees_scores: dictionary indicating subtress's labels and their respective current score.
        labels: (N,) flat vector of labels for each point. Represents the "cluster labels"
                for disconnected components of the graph.
        target_label: int from the labels array to subset the MSF. If passed along with `labels`, then a cut
                      will be found that is restricted to that subset of the MSF.
        make: bool, whether or not to modify the input MSF in order to make the best cut that was found.
        verbose: bool/int, denoting how much output to provide to the user, in terms
                 of print statements or progressbars

        Returns a namedtuple with in_node, out_node, and score.
        """
        if data is None:
            data = np.ones(MSF.shape)

        if (labels is None) != (target_label is None):
            raise ValueError(
                "Both labels and target_label must be supplied! Only {} provided.".format(
                    ["labels", "target_label"][int(target_label is None)]
                )
            )
        if verbose:
            try:
                from tqdm import tqdm
            except ImportError:

                def tqdm(noop, desc=""):
                    return noop

        else:

            def tqdm(noop, desc=""):
                return noop

        zero_in = (labels is not None) and (target_label is not None)
        best_deletion = deletion(np.nan, np.nan, np.inf)
        best_d_score = -np.inf

        try:
            if data_reg["reg"].__name__ == "GM_Lag" or data_reg["reg"].__name__ == "BaseGM_Lag":
                data_reg = self._prep_lag(data_reg)
        except:
            pass

        try:
            old_score = sum(trees_scores.values())
        except:
            pass
        best_scores = {}
        current_list = current_labels.tolist()
        for in_node, out_node in tqdm(
            np.vstack(MSF.nonzero()).T, desc="finding cut..."
        ):  # iterate over MSF edges
            if zero_in:
                if labels[in_node] != target_label:
                    continue

            local_MSF = copy.deepcopy(MSF)
            # delete a candidate edge
            local_MSF[in_node, out_node] = 0
            local_MSF.eliminate_zeros()
            current_tree = current_labels[in_node]

            # get the connected components
            local_n_subtrees, local_labels = cg.connected_components(
                local_MSF, directed=False
            )

            if local_n_subtrees <= current_n_subtrees:
                raise Exception("Malformed MSF!")

            # compute the score of these components
            if model_family == "spreg":
                new_score, new_trees_scores = self.score_spreg(
                    data, data_reg, local_labels, quorum, current_labels, current_tree
                )
            elif model_family == "statsmodels":
                new_score, new_trees_scores = self.score_stats(
                    data, data_reg, local_labels, quorum, current_labels, current_tree
                )
            else:
                raise ValueError("Model family must be either spreg or statsmodels.")

            if np.isfinite(new_score):
                try:
                    d_score = trees_scores[current_tree] - new_score
                    score = old_score - d_score
                except:
                    d_score = -new_score
                    score = new_score
                # if the d_score is greater than the best score and quorum is met
                if d_score > best_d_score:
                    best_deletion = deletion(in_node, out_node, score)
                    best_d_score = d_score
                    try:
                        for i in set(current_labels):
                            best_scores[
                                local_labels[current_list.index(i)]
                            ] = trees_scores[i]
                        for i in new_trees_scores:
                            best_scores[i] = new_trees_scores[i]
                    except:
                        best_scores = new_trees_scores
                    best_MSF = local_MSF
                    best_labels = local_labels
        try:
            return best_deletion, best_scores, best_MSF, local_n_subtrees, best_labels
        except UnboundLocalError:  # in case no solution is found
            return deletion(None, None, np.inf), np.inf, None, np.inf, None


def _const_x(x):
    x = x[:, np.ptp(x, axis=0) != 0]
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    return x