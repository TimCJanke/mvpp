import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acovf
import warnings
import pandas as pd
from scipy.stats import rankdata
from statsmodels.distributions.empirical_distribution import ECDF


########## univariate scores ###########
def calibration_score(pseudo_obs, n_grid=100):
    # computes the L1 distance between the empirical CDF of the pseudo_obs
    # and the analytical values of the uniform CDF on a equidistant grid
    grid = np.linspace(0.0, 1.0, n_grid+1)[1:-1]
    ecdf_pobs = ECDF(np.ravel(pseudo_obs))
    return np.mean(np.abs(ecdf_pobs(grid) - grid))


# Pinball Score
def pinball_score(y, dat, taus, return_single_scores=False):

    """
    Compute average Pinball Score from quantiles of the predictive distribution.

    Parameters
    ----------
    y : array, shape (N,)
        True values.
    dat : array, shape (N,n_taus)
        Predicted quantiles.
    taus : array, shape (n_taus,)
        Quantiles to evaluate.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.
    return_qloss : bool, optional
        Return average scores for single quantiles. The default is False.

    Returns
    -------
    float or tuple.
        Default is to return mean pinball score.
        If return_single_scores is True, also returns array of single scores of shape (N,).
        If return_qloss is True, also returns array of average pinball scores per quantile of shape (n_taus,).

    """
    
    if len(y.shape) == 1:
        y = np.expand_dims(y,1)
    taus = np.asarray(taus)
    err = y-dat
    q_loss = np.maximum(err*taus,err*(taus-1))
    
    if return_single_scores is True:
        return q_loss
    else:
        return np.mean(q_loss)



# Continuous Ranked Probability Score (CRPS)
def crps_sample(y_true, y_pred, return_single_scores=False):
    """
    Compute Continuous Ranked Probability Score (CRPS) from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape(n_examples,)
        True values.
    dat : array, shape (n_examples, n_samples)
        Predictive scenarios.

    Returns
    -------
    float or tuple of (float, array)
        Returns average CRPS.
        If return_single_scores is True also returns array of scores for single examples.

    """

    if len(y_true.shape)==1:
        y_true = np.expand_dims(y_true, 1)

    crps_1 = np.mean(np.abs(y_true - y_pred), axis=1)
    crps_2 = np.zeros(y_true.shape[0])
    for i in range(y_pred.shape[0]):
        crps_2[i] = np.mean(np.abs(y_pred[[i],:].T - np.repeat(y_pred[[i],:], y_pred.shape[1], axis=0)),)

    scores = crps_1 - 0.5*crps_2
    if return_single_scores:
        return scores
    else:
        return np.mean(scores)




########## multivariate scores ###########
   
def calibration_score_sample(y_true, y_pred, n_grid=100):
    """Computes the calibration score from true values and predicted samples.

    Args:
        y_true (np.array): An array of shape (N,D) containing the observed values for each dimesnion
        y_pred (np.array): An array of shape (N,D,M) containing M possible realizations per observation and dimension
        n_grid (int, optional): Number of equal width bins to compute histogram values. Defaults to 100.

    Returns:
        float: calibration score averaged over dimesnions
    """
    
    if len(y_true.shape)==1:
        y_true = np.expand_dims(y_true, 1)

    if len(y_pred.shape)==2:
        y_pred = np.expand_dims(y_pred, 1)

    pobs_true, _ = _get_pobs(y_true, y_pred, make_uniform=False)

    scores = []
    for d in range(y_true.shape[1]):
        scores.append(calibration_score(pobs_true[:, d], n_grid=n_grid))
    
    return np.mean(scores)
    
    

# Energy Score
def es_sample(y_true, y_pred, return_single_scores=False):
    """
    Compute mean energy score from samples of the predictive distribution.

    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.

    Returns
    -------
    float or tuple of (float, array)
        Mean energy score. If return_single_scores is True also returns scores for single examples.

    """
    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must contain same number of examples."
    assert y_true.shape[1] == y_pred.shape[1], "Examples in y_true and y_pred must have same dimension."


    N = y_true.shape[0]
    M = y_pred.shape[2]

    es_12 = np.zeros(y_true.shape[0])
    es_22 = np.zeros(y_true.shape[0])

    for i in range(N):
        es_12[i] = np.sum(np.sqrt(np.sum(np.square((y_true[[i],:].T - y_pred[i,:,:])), axis=0)))
        es_22[i] = np.sum(np.sqrt(np.sum(np.square(np.expand_dims(y_pred[i,:,:], axis=2) - np.expand_dims(y_pred[i,:,:], axis=1)), axis=0)))
    
    scores = es_12/M - 0.5* 1/(M*M) * es_22
    if return_single_scores:
        return scores
    else:
        return np.mean(scores)



# Variogram Score
def vs_sample(y_true, y_pred, p=0.5, return_single_scores=False):
    """
    Compute mean variogram score from samples of the predictive distribution. 
    
    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    p : float, optional
        Order of variagram score. The default is 0.5.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.
    
    Returns
    -------
    float or tuple of (float, array)
        Average variogram score. If return_single_scores is True also returns scores for single examples.

    """
    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must contain same number of examples."
    assert y_true.shape[1] == y_pred.shape[1], "Examples in y_true and y_pred must have same dimension."

    N = y_true.shape[0]
    D = y_true.shape[1]
    M = y_pred.shape[2]


    scores = np.zeros(y_true.shape[0])
    for i in range(N):
        
        vs_1 = np.power(np.abs(y_true[[i],:].T - y_true[[i],:]), p)
        vs_2 = np.mean(np.power(np.abs(np.repeat(y_pred[i,:,:], repeats=D, axis=0) - np.tile(y_pred[i,:,:], (D,1))), p), axis=1)
        scores[i] = np.sum(np.square(np.ravel(vs_1) - vs_2))
    
    if return_single_scores:
        return scores
    else:
        return np.mean(scores)



def all_scores_mv_sample(y_true, y_pred, 
                          return_single_scores = False,
                          taus=[0.1, 0.9],
                          CALIBRATION=True,
                           MSE=True, MAE=True, 
                           PB=True, CRPS=True, ES=True, VS05=True, VS1=False, 
                           CES=False, CVS05=False, CVS1=False):
    """
    Compute an set of univariate and multivariate scoring rules based on samples.

    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution. 
    CALIBRATION : TYPE, optional
        DESCRIPTION. The default is True.    
    MSE : TYPE, optional
        DESCRIPTION. The default is True.
    MAE : TYPE, optional
        DESCRIPTION. The default is True.
    CRPS : TYPE, optional
        DESCRIPTION. The default is True.
    ES : TYPE, optional
        DESCRIPTION. The default is True.
    VS05 : TYPE, optional
        DESCRIPTION. The default is True.
    VS1 : TYPE, optional
        DESCRIPTION. The default is True.
    CES : TYPE, optional
        DESCRIPTION. The default is True.
    CVS05 : TYPE, optional
        DESCRIPTION. The default is True.
    CVS1 : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    scores : TYPE
        DESCRIPTION.

    """

    assert len(y_pred.shape) == 3, "y_pred must be a three dimesnional array of shape (n_examples, n_dim, n_samples)"
    assert len(y_true.shape) == 2, "y_true must be a two dimesnional array of shape (n_examples, n_dim)"
    
    assert y_true.shape[0] == y_pred.shape[0], "y_true and y_pred must contain same number of examples."
    assert y_true.shape[1] == y_pred.shape[1], "Examples in y_true and y_dat must have same dimension."

    
    scores = {}
    
    if CALIBRATION:
        scores["CAL"] = calibration_score_sample(y_true, y_pred)
        
    if MSE:
        s = np.square(y_true - np.mean(y_pred,axis=2))
        if return_single_scores: 
            scores["MSE"] = np.mean(s, axis=1)
        else:
            scores["MSE"] = np.mean(s)
    
    if MAE:
        s = np.abs(y_true - np.median(y_pred,axis=2))
        if return_single_scores: 
            scores["MAE"] = np.mean(s, axis=1)
        else:
            scores["MAE"] = np.mean(s)    
    
    if PB:
        q_pred = np.transpose(np.quantile(y_pred, q=taus, axis=2, method="linear"), (1,2,0)) # [N, D, len(taus)]
        s = pinball_score(np.reshape(y_true, (-1,1)), np.reshape(q_pred, (-1, len(taus))), taus=taus, return_single_scores=True) # [NxD,len(taus)]
        for i, tau in enumerate(taus):
            if return_single_scores:
                scores["PB_"+str(tau)] = np.mean(np.reshape(s[:,i], (-1, y_true.shape[1])), axis=1)
            else:
                scores["PB_"+str(tau)] = np.mean(s[:,i])
    
    if CRPS:
        s = np.reshape(crps_sample(np.reshape(y_true, (-1,1)), 
                                                np.reshape(y_pred, (-1, y_pred.shape[2])), 
                                                return_single_scores=True),
                       (-1, y_true.shape[1]))
        
        if return_single_scores:
            scores["CRPS"] = np.mean(np.reshape(s, (-1, y_true.shape[1])), axis=1)
        else:
            scores["CRPS"] = np.mean(s)
    
    if ES:
        scores["ES"] = es_sample(y_true, y_pred, return_single_scores=return_single_scores)
    
    if VS05:
        scores["VS05"] = vs_sample(y_true, y_pred, p=0.5, return_single_scores=return_single_scores)
    
    if VS1:
        scores["VS1"] = vs_sample(y_true, y_pred, p=1.0, return_single_scores=return_single_scores)
        
    if CES or CVS05 or CVS1:     
        
        y_true_pobs, y_predict_pobs  = _get_pobs(y_true, y_pred, make_uniform=True)
        
        if CES:
            scores["CES"] = 1/np.sqrt(y_pred.shape[1]) * es_sample(y_true_pobs, y_predict_pobs, return_single_scores=return_single_scores) - (0.25 - 0.5*(1/np.sqrt(6)))
        
        if CVS05:
            scores["CVS05"] = vs_sample(y_true_pobs, y_predict_pobs, p=0.5, return_single_scores=return_single_scores)
        
        if CVS1:
            scores["CVS1"] = vs_sample(y_true_pobs, y_predict_pobs, p=1.0, return_single_scores=return_single_scores)
        
    return scores
    

def _get_pobs(y, dat, make_uniform=False):
    """ Obtain pseudo observations for data as well as for y under the distribution represented by samples in dat[i,d,:]."""
    N,D,M = dat.shape
    
    # rank data in each sampled scenario
    pobs_dat = np.zeros_like(dat)
    for n in range(N):
        for d in range(D):
            pobs_dat[n,d,:] = (2*_rank_data_random_tiebreaker(dat[n,d,:])-1)/(2*M)
            
    idx = np.argmin(np.abs(np.expand_dims(y, axis=2) - dat), axis=2) # return index of nearest rank for observed y for each i and d
    pobs_y = np.take_along_axis(pobs_dat, indices=np.expand_dims(idx, axis=2), axis=2)[:,:,0] # select nearest rank
    
    if make_uniform is True:
    # ensure uniformity for uncalibrated predictive distributions
        pobs_y_adjusted = np.zeros_like(pobs_y)
        for d in range(D):
            pobs_y_adjusted[:,d] = (2*_rank_data_random_tiebreaker(pobs_y[:,d])-1)/(2*N)
    
        return pobs_y_adjusted, pobs_dat
    else:
        return pobs_y, pobs_dat
        


def _rank_data_random_tiebreaker(a):
    #""" Ranks data in 1d array using ordinal ranking (i.e. all ranks are assigned) and breaks ties at random """
    
    idx = np.random.permutation(len(a)) # indexes for random shuffle
    
    ranks_shuffled = rankdata(a[idx], method="ordinal") # compute ranks of shuffled index
    
    idx_r = np.zeros_like(idx) # indexes to invert permutation
    idx_r[idx] = np.arange(len(idx))
    
    ranks = ranks_shuffled[idx_r] # restore original order
    
    return ranks



def dm_test(loss_1, loss_2, h=1, hln_correction=True):
    """DM test with optional HLN correction (as in R forecast package)
        Tests the null hypothesis that forecasts from both models have the same accuracy,
        the alternative hypothesis is that model 1 has better accuracy than model 2.
    Args:
        loss_1 (array): loss series of model 1
        loss_2 (array): loss series of model 2
        h (int, optional): Steps ahead. Defaults to 1.
        hln_correction (bool, optional): If HLN correction is used. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        float: p value from DM test
    """
    
    diff = loss_1 - loss_2 # loss differential
    N = len(diff) # size of data set

    if hln_correction:
        # Harvey Granger Newbold version
        auto_cov = acovf(diff, nlag=h-1, missing="raise")
        var = (auto_cov[0] + 2*np.sum(auto_cov[1:]))
        k = np.sqrt(( N + 1 - 2*h + h*(h-1)/N)/N)
        dist = stats.t(df=N-1)
    else:
        # vanilla DM test
        var = np.var(diff, ddof=0)
        k = 1.0
        dist = stats.norm()

    # auto covariance / variance can be negative in some cases
    if var > 0:
        dm_statistic = np.mean(diff)/np.sqrt(var/N) # dm statistic
    elif h==1:
        raise ValueError("Variance of DM statistic is zero")
    else:
        warnings.warn("Variance is negative, using horizon h=1.")
        return dm_test(loss_1, loss_2, h=1, hln_correction=hln_correction)
        
    p_value = dist.cdf(dm_statistic*k) # compute p-value
    
    return p_value


def dm_test_matrix(losses: dict, h=1, hln_correction=True, model_names=None):
    """_summary_

    Args:
        losses (dict): Dictionary of loss series where keys are model names
        h (int, optional): Used for DM test, how many steps ahead is forecasted. Defaults to 1.
        hln_correction (bool, optional): If hln correction should be used in DM test. Defaults to True.
        model_names (_type_, optional): Alternative names for models, if None use keys in losses. Defaults to None.

    Returns:
        pandas DataFrame: Dataframe of pairwise DM test p-values
    """
    if model_names is None:
        model_names = losses.keys()
    
    p_val_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=np.float)
    for i, m_i in enumerate(losses):
        for j, m_j in enumerate(losses):
            if i !=j:
                p_val_matrix.loc[m_i,m_j] = dm_test(losses[m_i], losses[m_j], h=h, hln_correction=hln_correction)
    return p_val_matrix.T