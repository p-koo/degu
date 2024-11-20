from scipy import stats
from sklearn import metrics


def eval_regression(pred, y, verbose=1):
    """Evaluate regression model performance.
    
    Args:
        pred: Model predictions
        y: True values
        
    Returns:
        list: Performance metrics [MSE, Pearson, Spearman] per task
    """
    num_tasks = y.shape[1]
    results = []
    for i in range(num_tasks):
        mse = metrics.mean_squared_error(y[:,i], pred[:,i])
        pearsonr = stats.pearsonr(y[:,i], pred[:,i])[0]
        spearmanr = stats.spearmanr(y[:,i], pred[:,i])[0]
        results.append([mse, pearsonr, spearmanr])
        if verbose:
            print('Task %d  MSE      = %.4f'%(i, mse))
            print('Task %d  Pearson  = %.4f'%(i, pearsonr))
            print('Task %d  Spearman = %.4f'%(i, spearmanr))
    return results


def eval_classification(pred, y, verbose=1):
    """Evaluate classification model performance.
    
    Args:
        pred: Model predictions
        y: True labels
        
    Returns:
        list: Performance metrics [AUROC, AUPR, F1] per task
    """
    num_tasks = y.shape[1]
    results = []
    for i in range(num_tasks):
        auroc = metrics.roc_auc_score(y[:,i], pred[:,i])
        aupr = metrics.average_precision_score(y[:,i], pred[:,i])  
        f1_score = metrics.f1_score(y[:,i], pred[:,i])  
        results.append([auroc, aupr, f1_score])
        if verbose:
            print('Task %d  AUROC = %.4f'%(i, auroc))
            print('Task %d  AUPR  = %.4f'%(i, aupr))
            print('Task %d  F1    = %.4f'%(i, f1_score))
    return results





