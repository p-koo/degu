import tensorflow as tf 

#-----------------------------------------------------------------------------
# Useful functions
#-----------------------------------------------------------------------------

def logvar(x, axis=0):
    """Calculate log variance along specified axis.
    
    Args:
        x: Input tensor
        axis: Axis to reduce along
        
    Returns:
        Log of variance
    """
    return tf.math.log(tf.math.reduce_variance(x, axis=axis))


def std(x, axis=0):
    """Calculate standard deviation along specified axis.
    
    Args:
        x: Input tensor
        axis: Axis to reduce along
        
    Returns:
        Standard deviation
    """
    return tf.math.reduce_std(x, axis=axis)


