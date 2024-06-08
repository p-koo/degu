import h5py
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

# create functions
def summary_statistics(pred, y, index):
    mse = mean_squared_error(y[:,index], pred[:,index])
    pearsonr = stats.pearsonr(y[:,index], pred[:,index])[0]
    spearmanr = stats.spearmanr(y[:,index], pred[:,index])[0]
    print(' MSE task ' + str(index) + ' = ' + str("{0:0.4f}".format(mse)))
    print(' PCC task ' + str(index) + ' = ' + str("{0:0.4f}".format(pearsonr)))
    print(' SCC task ' + str(index) + ' = ' + str("{0:0.4f}".format(spearmanr)))
    return mse, pearsonr, spearmanr 


def load_deepstarr(filepath):
    dataset = h5py.File('deepstarr_data.h5', 'r')
    x_train = np.array(dataset['x_train']).astype(np.float32)
    y_train = np.array(dataset['y_train']).astype(np.float32).transpose()
    x_valid = np.array(dataset['x_valid']).astype(np.float32)
    y_valid = np.array(dataset['y_valid']).astype(np.float32).transpose()
    x_test = np.array(dataset['x_test']).astype(np.float32)
    y_test = np.array(dataset['y_test']).astype(np.float32).transpose()
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def downsample_trainset(x_train, y_train, downsample_frac, seed=12345):
    N = x_train.shape[0]
    np.random.seed(seed)
    num_downsample = int(N*downsample_frac)
    return x_train[:num_downsample], y_train[:num_downsample]


################################################################
# Uncertainty evaluation metrics
################################################################

def gaussian_confidence_interval(mean, std, alpha=0.05):

    # Calculate the z-score for the given confidence level
    z_score = stats.norm.ppf(1 - (alpha / 2))

    # Calculate the margin of error
    margin_of_error = z_score * std

    # Calculate the lower and upper bounds
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    return lower_bound, upper_bound


def laplace_confidence_interval(mu, b, alpha=0.95):

    quantile = -np.log(1 - alpha)

    # Calculate the margin of error
    margin_of_error = quantile * b

    # Calculate the lower and upper bounds
    lower_bound = mu - margin_of_error
    upper_bound = mu + margin_of_error

    return lower_bound, upper_bound


def cauchy_confidence_interval(mu, b, alpha=0.95):

    quantile = np.tan(np.pi * (1 - alpha / 2))

    # Calculate the margin of error
    margin_of_error = quantile * b

    # Calculate the lower and upper bounds
    lower_bound = mu - margin_of_error
    upper_bound = mu + margin_of_error

    return (lower_bound, upper_bound)


def prediction_interval_coverage_probability(y, lower_bound, upper_bound):
    within_ci = sum(y[i] >= lower_bound[i] and y[i] <= upper_bound[i] for i in range(len(y)))    
    fraction = (within_ci / len(y)) 
    return fraction


def average_interval_length(upper_bound, lower_bound):
    interval_len = upper_bound - lower_bound
    avg_interval = np.mean(interval_len)
    std_interval = np.std(interval_len)
    return avg_interval, std_interval




################################################################
# MC dropout
################################################################
'''
prerequisite:
    - model 
    - x_test

example:
    mc_model = ModelFun(...)
    mc_model.set_weights(model.get_weights())
    mc_std = mc_dropout_std(mc_model, x_test, mc_dropout_rate=0.25, num_samples=100, batch_size=512)
'''

def predict_in_batches(model, x, batch_size=32, training=True):
    N = x.shape[0]
    predictions = []
    for start_idx in range(0, N, batch_size):
        end_idx = start_idx + batch_size
        batch_preds = model(x[start_idx:end_idx], training=training)
        predictions.append(batch_preds)
    return np.concatenate(predictions, axis=0)



def mc_dropout_std(model, x, mc_dropout_rate=0.2, num_samples=100, batch_size=512):

    # Initialize an empty list to store predictions
    preds = []
    for i in range(num_samples):
        preds.append(predict_in_batches(model, x, batch_size, training=True))

    # Convert the predictions to a NumPy array
    preds = np.array(preds)

    # Calculate the standard deviation of predictions along the sample dimension
    pred_std = np.std(preds, axis=0)
    pred_mean = np.mean(preds, axis=0)
    
    return pred_mean, pred_std




#############################################################################################
# Attribution consistency
#############################################################################################
'''
prerequisite:
    - x_seq (generated or observed seqeunces with shapes (N,L,A)) 
    - oracle model 

example:
    shap_score = gradient_shap(x_seq, oracle, task_idx)
    attributino_map = process_attribution_map(shap_score)
    mask = unit_mask(x_seq)
    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map], x_seq, mask, radius_count_cutoff)
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
    entropic_information = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
'''

def process_attribution_map(saliency_map_raw):
    saliency_map_raw = saliency_map_raw - np.mean(saliency_map_raw, axis=-1, keepdims=True) # gradient correction
    saliency_map_raw = saliency_map_raw / np.sum(np.sqrt(np.sum(np.square(saliency_map_raw), axis=-1, keepdims=True)), axis=-2, keepdims=True) #normalize
    saliency_map_raw_rolled = np.roll(saliency_map_raw, -1, axis=-2)
    saliency_map_raw_rolled_twice = np.roll(saliency_map_raw, -2, axis=-2)
    saliency_map_raw_rolled_triple = np.roll(saliency_map_raw, -3, axis=-2)
    saliency_map_raw_rolled_4 = np.roll(saliency_map_raw, -4, axis=-2)
    saliency_map_raw_rolled_5 = np.roll(saliency_map_raw, -5, axis=-2)
    saliency_map_raw_rolled_6 = np.roll(saliency_map_raw, -6, axis=-2)
    # Define k-window here, include k terms below (here k = 3)
    saliency_special = saliency_map_raw + saliency_map_raw_rolled + saliency_map_raw_rolled_twice #+ saliency_map_raw_rolled_triple # + saliency_map_raw_rolled_4 + saliency_map_raw_rolled_5 #This line is optional.
    saliency_special = ortonormal_coordinates(saliency_special) #Down to 3D, since data lives on the plane.
    return saliency_special

def unit_mask(x_seq):
    return np.sum(np.ones(x_seq.shape),axis=-1) / 4

def spherical_coordinates_process_2_trad(saliency_map_raw_s, X, mask, radius_count_cutoff=0.04):
    global N_EXP
    N_EXP = len(saliency_map_raw_s)
    radius_count=int(radius_count_cutoff * np.prod(X.shape)/4)
    cutoff=[]
    x_s, y_s, z_s, r_s, phi_1_s, phi_2_s = [], [], [], [], [], []
    for s in range (0, N_EXP):
        saliency_map_raw = saliency_map_raw_s[s]
        xxx_motif=saliency_map_raw[:,:,0]
        yyy_motif=(saliency_map_raw[:,:,1])
        zzz_motif=(saliency_map_raw[:,:,2])
        xxx_motif_pattern=saliency_map_raw[:,:,0]*mask
        yyy_motif_pattern=(saliency_map_raw[:,:,1])*mask
        zzz_motif_pattern=(saliency_map_raw[:,:,2])*mask
        r=np.sqrt(xxx_motif*xxx_motif+yyy_motif*yyy_motif+zzz_motif*zzz_motif)
        resh = X.shape[0] * X.shape[1]
        x=np.array(xxx_motif_pattern.reshape(resh,))
        y=np.array(yyy_motif_pattern.reshape(resh,))
        z=np.array(zzz_motif_pattern.reshape(resh,))
        r=np.array(r.reshape(resh,))
        #Take care of any NANs.
        x=np.nan_to_num(x)
        y=np.nan_to_num(y)
        z=np.nan_to_num(z)
        r=np.nan_to_num(r)
        cutoff.append( np.sort(r)[-radius_count] )
        R_cuttof_index = np.sqrt(x*x+y*y+z*z) > cutoff[s]
        #Cut off
        x=x[R_cuttof_index]
        y=y[R_cuttof_index]
        z=z[R_cuttof_index]
        r=np.array(r[R_cuttof_index])
        x_s.append(x)
        y_s.append(y)
        z_s.append(z)
        r_s.append(r)
        #rotate axis
        x__ = np.array(y)
        y__ = np.array(z)
        z__ = np.array(x)
        x = x__
        y = y__
        z = z__
        #"phi"
        phi_1 = np.arctan(y/x) #default
        phi_1 = np.where((x<0) & (y>=0), np.arctan(y/x) + PI, phi_1)   #overwrite
        phi_1 = np.where((x<0) & (y<0), np.arctan(y/x) - PI, phi_1)   #overwrite
        phi_1 = np.where (x==0, PI/2, phi_1) #overwrite
        #Renormalize temorarily to have both angles in [0,PI]:
        phi_1 = phi_1/2 + PI/2
        #"theta"
        phi_2=np.arccos(z/r)
        #back to list
        phi_1 = list(phi_1)
        phi_2 = list(phi_2)
        phi_1_s.append(phi_1)
        phi_2_s.append(phi_2)
    #print(cutoff)
    return phi_1_s, phi_2_s, r_s

def initialize_integration_2(box_length):
    LIM = 3.1416
    global volume_border_correction
    box_volume = box_length*box_length
    n_bins = int(LIM/box_length)
    volume_border_correction =  (LIM/box_length/n_bins)*(LIM/box_length/n_bins)
    #print('volume_border_correction = ', volume_border_correction)
    n_bins_half = int(n_bins/2)
    return LIM, box_length, box_volume, n_bins, n_bins_half

def calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, box_length, box_volume, prior_range):
    global Empirical_box_pdf_s
    global Empirical_box_count_s
    global Empirical_box_count_plain_s
    Empirical_box_pdf_s=[]
    Empirical_box_count_s = []
    Empirical_box_count_plain_s = []
    prior_correction_s = []
    Spherical_box_prior_pdf_s=[]
    for s in range (0,N_EXP):
        #print(s)
        Empirical_box_pdf_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[0])
        Empirical_box_count_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[1])
        Empirical_box_count_plain_s.append(Empiciral_box_pdf_func_2(phi_1_s[s],phi_2_s[s], r_s[s], n_bins, box_length, box_volume)[2])
    Entropic_information = []
    for s in range (0,N_EXP):
        Entropic_information.append ( KL_divergence_2 (Empirical_box_pdf_s[s], Empirical_box_count_s[s], Empirical_box_count_plain_s[s], n_bins, box_volume, prior_range)  )
    return list(Entropic_information)

def KL_divergence_2(Empirical_box_pdf, Empirical_box_count, Empirical_box_count_plain, n_bins, box_volume, prior_range):  #, correction2)
    # p= empirical distribution, q=prior spherical distribution
    # Notice that the prior distribution is never 0! So it is safe to divide by q.
    # L'Hospital rule provides that p*log(p) --> 0 when p->0. When we encounter p=0, we would just set the contribution of that term to 0, i.e. ignore it in the sum.
    Relative_entropy = 0
    PI = 3.1416
    for i in range (1, n_bins-1):
        for j in range(1,n_bins-1):
            if (Empirical_box_pdf[i,j] > 0  ):
                phi_1 = i/n_bins*PI
                phi_2 = j/n_bins*PI
                correction3 = 0
                prior_counter = 0
                prior=0
                for ii in range(-prior_range,prior_range):
                    for jj in range(-prior_range,prior_range):
                        if(i+ii>0 and i+ii<n_bins and j+jj>0 and j+jj<n_bins):
                            prior+=Empirical_box_pdf[i+ii,j+jj]
                            prior_counter+=1
                prior=prior/prior_counter
                if(prior>0) : KL_divergence_contribution = Empirical_box_pdf[i,j] * np.log (Empirical_box_pdf[i,j]  /  prior )
                if(np.sin(phi_1)>0 and prior>0 ): Relative_entropy+=KL_divergence_contribution  #and Empirical_box_count_plain[i,j]>1
    Relative_entropy = Relative_entropy * box_volume #(volume differential in the "integral")
    return np.round(Relative_entropy,3)

def Empiciral_box_pdf_func_2 (phi_1, phi_2, r_s, n_bins, box_length, box_volume):
    N_points = len(phi_1) #Number of points
    Empirical_box_count = np.zeros((n_bins, n_bins))
    Empirical_box_count_plain = np.zeros((n_bins, n_bins))
    #Now populate the box. Go over every single point.
    for i in range (0, N_points):
        # k, l are box numbers of the (phi_1, phi_2) point
        k=np.minimum(int(phi_1[i]/box_length), n_bins-1)
        l=np.minimum(int(phi_2[i]/box_length), n_bins-1)
        #Increment count in (k,l,m) box:
        Empirical_box_count[k,l]+=1*r_s[i]*r_s[i]
        Empirical_box_count_plain[k,l]+=1
    #To get the probability distribution, divide the Empirical_box_count by the total number of points.
    Empirical_box_pdf = Empirical_box_count / N_points / box_volume
    #Check that it integrates to around 1:
    #print('Integral of the empirical_box_pdf (before first renormalization) = ' , np.sum(Empirical_box_pdf*box_volume), '(should be 1.0 if OK) \n')
    correction = 1 / np.sum(Empirical_box_pdf*box_volume)
    #Another, optional correction 
    count_empty_boxes = 0
    count_single_points = 0
    for k in range (1, n_bins-1):
        for l in range(1,n_bins-1):
            if(Empirical_box_count[k,l] ==1):
                count_empty_boxes+=1
                count_single_points+=1
    return Empirical_box_pdf * correction * 1 , Empirical_box_count *correction , Empirical_box_count_plain #, correction2



