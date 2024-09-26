import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import norm, uniform, beta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



# generate noisy uniform examples (setting 1)
def generate_noisy_examples_U(N, alpha, bias = 'up', mag = 0.1):
    """
    Bias can be either up, no, or down, meaning how decision maker misunderstands the population's qualifications
    mag means how the estimated Y|X changes
    """
    if bias == 'up':
        term = mag
    else:
        term = -mag
    
    X1 = uniform.rvs(size = N)
    X2 = uniform.rvs(size = N)
    prob = 0.5*(X1+X2)+term


    y = uniform.rvs(size = N)
    y[y < prob] = 1
    y[y != 1] = -1

    X = np.vstack((X1,X2)).T
    y_noise = y
    return X, y_noise

# Function to generate N true uniform examples (setting 1)
def generate_true_examples_U(N, alpha):
    """
    Generate N true examples
    """
    X1 = uniform.rvs(size = N)
    X2 = uniform.rvs(size = N)
    prob = 0.5*(X1+X2)
    y = uniform.rvs(size = N)
    y[y < prob] = 1
    y[y != 1] = -1

    X = np.vstack((X1,X2)).T
    return X, y

# Function to generate N noisy Gaussian examples (setting 2)
def generate_noisy_examples_G(N, alpha, bias = 'up', mag = 0.1, loc = 1, scale = 0.5, multi=False):
    """
    Bias can be either up, no, or down, meaning how decision maker misunderstands the population's qualifications
    mag means how the estimated Y|X changes
    """
    if bias == 'up':
        term = mag
    else:
        term = -mag
    
    X1 = norm.rvs(0,0.5,size=N)
    X2 = norm.rvs(0,0.5,size=N)
    prob = 1/(1+np.exp(X1+X2))+term
    y = uniform.rvs(size = N)
    y[y < prob] = 1
    y[y != 1] = -1

    X = np.vstack((X1,X2)).T
    y_noise = y
    return X, y_noise


# Function to generate N true Gaussian examples (setting 2)
def generate_true_examples_G(N, alpha, loc = 1, scale = 0.5, multi=False):
    """
    Generate N true examples
    """
    X1 = norm.rvs(0,0.5,size=N)
    X2 = norm.rvs(0,0.5,size=N)
    prob = 1/(1+np.exp(X1+X2))
    y = uniform.rvs(size = N)
    y[y < prob] = 1
    y[y != 1] = -1

    X = np.vstack((X1,X2)).T

    return X, y


# Generate real examples with fitted Beta distributions
def generate_noisy_examples_R(N, alphas, params_1, params_2, bias = 'up', mag = 0.1, group = 'a'):
    """
    Bias can be either up, no, or down, meaning how decision maker misunderstands the population's qualifications
    mag means how the estimated qualification rate changes
    """
    alpha = alphas[group]
    if bias == 'up':
        N1 = int(N*(alpha+mag))
    else:
        N1 = int(N*(alpha-mag))
    N2 = N - N1
    X11 = beta.rvs(params_1[group]['p'][0],params_1[group]['p'][1], loc = 0, scale = 1, size = N1)
    X12 = beta.rvs(params_2[group]['p'][0],params_2[group]['p'][1], loc = 0, scale = 1, size = N1)
    X21 = beta.rvs(params_1[group]['n'][0],params_1[group]['n'][1], loc = 0, scale = 1, size = N2)
    X22 = beta.rvs(params_2[group]['n'][0],params_1[group]['n'][1], loc = 0, scale = 1, size = N2)
    X1 = np.vstack((X11,X12)).T
    X2 = np.vstack((X21,X22)).T
    y1 = np.array(N1*[1])
    y2= np.array(N2*[-1])
    X = np.concatenate((X1,X2))
    y = np.concatenate([y1,y2])
    y_noise = y
    
    return X, y_noise


# Generate real examples with fitted Beta distributions
def generate_noisy_examples_R2(N, alphas, params_1, params_2, bias = 'up', mag = 0.1, group = 'a'):
    """
    Bias can be either up, no, or down, meaning how decision maker misunderstands the population's qualifications
    mag means how the estimated qualification rate changes
    """
    alpha = alphas[group]
    if bias == 'up':
        N1 = int(N*(alpha+mag))
    else:
        N1 = int(N*(alpha-mag))
    N2 = N - N1
    X11 = beta.rvs(params_1[group]['p'][0],params_1[group]['p'][1], loc = 0.3, scale = 0.3, size = N1)
    X12 = beta.rvs(params_2[group]['p'][0],params_2[group]['p'][1], loc = 0.55, scale = 0.35, size = N1)
    X21 = beta.rvs(params_1[group]['n'][0],params_1[group]['n'][1], loc = 0.3, scale = 0.3, size = N2)
    X22 = beta.rvs(params_2[group]['n'][0],params_1[group]['n'][1], loc = 0.55, scale = 0.35, size = N2)
    X1 = np.vstack((X11,X12)).T
    X2 = np.vstack((X21,X22)).T
    y1 = np.array(N1*[1])
    y2= np.array(N2*[-1])
    X = np.concatenate((X1,X2))
    y = np.concatenate([y1,y2])
    y_noise = y
    
    return X, y_noise


# Function to generate N true examples
def generate_true_examples_R(N,alphas,params_1, params_2, group = 'a'):
    """
    Generate N true examples
    """
    alpha = alphas[group]
    N1 = int(N*alpha)
    N2 = N - N1
    X11 = beta.rvs(params_1[group]['p'][0],params_1[group]['p'][1], loc = 0, scale = 1, size = N1)
    X12 = beta.rvs(params_2[group]['p'][0],params_2[group]['p'][1], loc = 0, scale = 1, size = N1)
    X21 = beta.rvs(params_1[group]['n'][0],params_1[group]['n'][1], loc = 0, scale = 1, size = N2)
    X22 = beta.rvs(params_2[group]['n'][0],params_1[group]['n'][1], loc = 0, scale = 1, size = N2)
    X1 = np.vstack((X11,X12)).T
    X2 = np.vstack((X21,X22)).T
    y1 = np.array(N1*[1])
    y2= np.array(N2*[-1])
    X = np.concatenate((X1,X2))
    y = np.concatenate([y1,y2])
    return X, y


# Function to generate N true examples
def generate_true_examples_R2(N,alphas,params_1, params_2, group = 'a'):
    """
    Generate N true examples
    """
    alpha = alphas[group]
    N1 = int(N*alpha)
    N2 = N - N1
    X11 = beta.rvs(params_1[group]['p'][0],params_1[group]['p'][1], loc = 0.3, scale = 0.3, size = N1)
    X12 = beta.rvs(params_2[group]['p'][0],params_2[group]['p'][1], loc = 0.55, scale = 0.35, size = N1)
    X21 = beta.rvs(params_1[group]['n'][0],params_1[group]['n'][1], loc = 0.3, scale = 0.3, size = N2)
    X22 = beta.rvs(params_2[group]['n'][0],params_1[group]['n'][1], loc = 0.55, scale = 0.35, size = N2)
    X1 = np.vstack((X11,X12)).T
    X2 = np.vstack((X21,X22)).T
    y1 = np.array(N1*[1])
    y2= np.array(N2*[-1])
    X = np.concatenate((X1,X2))
    y = np.concatenate([y1,y2])
    return X, y


# Function to generate KDE noisy
def generate_noisy_examples_K(N, clf, kde_x, bias = 'up', mag = 0.1):
    """
    Bias can be either up, no, or down, meaning how decision maker misunderstands the population's qualifications
    mag means how the estimated qualification rate changes
    """
    if N == 0:
        return np.array([]),np.array([]),np.array([])
    X_full = kde_x.sample(N)
    X = X_full[:,:10]
    y_raw = clf.predict_proba(X_full)[:,1]
    if bias == 'up':
        y_raw += mag
    else:
        y_raw -= mag
    
    sample = uniform.rvs(size=N)
    sample[sample<y_raw]=1
    sample[sample<1]=0
    y_noise = sample
    
    return X, y_noise, X_full

def generate_true_examples_K(N, clf, kde_x):
    """
    True has no bias
    """
    if N == 0:
        return np.array([]),np.array([]),np.array([])
    X_full = kde_x.sample(N)
    X = X_full[:,:10]
    y_raw = clf.predict_proba(X_full)[:,1]
    sample = uniform.rvs(size=N)
    sample[sample<y_raw]=1
    sample[sample<1]=0
    y = sample
    
    return X, y, X_full

# function for best response
def best_response(w, b, Q, x):
    """
    Given a classifier specified by (w,b), a cost matrix Q,  and a point x, 
    output its position after best response.
    w: d*1 vector
    Q: d*d vector
    b: real number
    x: d*1 vector
    First calculate the easiest way to move to decision boundary,
    then check whether the response is cost effective, if not just return x.
    Otherwise, return x+dx
    """
    from numpy.linalg import inv
    B = -(np.matmul(w.T, x)+b).item()
    # This means x is already admitted
    if B < 0:
        return x
    # dx solved by calculating KKT conditions
    D = np.matmul(w.T, np.matmul(inv(Q), w)).item()
    dx = B*np.matmul(inv(Q), w)/D
    dx = dx.reshape(-1)
    cost = np.matmul(dx.T, np.matmul(Q, dx)).item()
    # not cost effective
    if cost > 2:
        return x
    return x+dx

# function for noisy best response
def noisy_response(w, b, Q, x, sd = 0.1):
    """
    Given a classifier specified by (w,b), a cost matrix Q,  and a point x, 
    output its position after best response.
    w: d*1 vector
    Q: d*d vector
    b: real number
    x: d*1 vector
    First calculate the easiest way to move to decision boundary,
    then check whether the response is cost effective, if not just return x.
    Otherwise, return x+dx
    """
    from numpy.linalg import inv
    eps = np.random.normal(loc = 0, scale=sd)
    B = -(np.matmul(w.T, x)+b).item()+eps
    # This means x is already admitted
    if B < 0:
        return x
    # dx solved by calculating KKT conditions
    D = np.matmul(w.T, np.matmul(inv(Q), w)).item()
    dx = B*np.matmul(inv(Q), w)/D
    dx = dx.reshape(-1)
    cost = np.matmul(dx.T, np.matmul(Q, dx)).item()
    # not cost effective
    if cost > 2:
        return x
    return x+dx

# function to return a SGD classifier using log loss
def get_classifier(X,y,a=0):
    """
    Output the true Logistic classifier using true data
    """
    if a == 0:
        clf_dis = SGDClassifier(loss = 'log_loss', max_iter=10000, tol=1e-6, penalty = None)
    else:
        # real data
        clf_dis = SGDClassifier(loss = 'log_loss', max_iter=10000, tol=1e-6, penalty = 'l2',alpha=a)
    clf_dis.fit(X,y)
    return clf_dis

# function to plot points
def plot_points(X, y, ax):
    X11 = X[y == 1][:,0]
    X12 = X[y == 1][:,1]
    X21 = X[y == -1][:,0]
    X22 = X[y == -1][:,1]
    ax.plot(X11, X12, 'k.', color = 'blue', marker = 'o', markersize=1, label = 'qualified')
    ax.plot(X21, X22, 'k.', color='red', marker='s', markersize=1, label = 'unqualified')

# Function to plot any classifier
def plot_classifier(w,b,label,color,ax):
    x = np.linspace(-1,2,100)
    boundary = [-(w[0][0]/w[0][1])*xi - b/w[0][1] for xi in x] 
    ax.plot(x, boundary, color = color, label = label, lw = 2)


# Function to generate strategic examples along with their true labels
def strategic_examples(Xt, yt_true, y_previous_pred, w, b, Q, tp = 0, alpha = 0.5, loc = 0, scale = 0.5, params_1 = {}, params_2 = {}, group = 'a', noisy=0, kde_x=None, clf=None, Xt_full=None):
    """
    Get new examples after strategic behaviors
    """
    # Strategic improvement and label change
    w = w.T.reshape(len(w[0]),1)
    for i in range(len(Xt)):
        # only unadmitted ones will improve
        if y_previous_pred[i] == 1:
            continue
        # calculate the position of the new feature
        if noisy > 0:
            xnew = noisy_response(w,b,Q,Xt[i],sd=noisy)
        else:
            xnew = best_response(w, b, Q, Xt[i])
        Xt[i] = xnew

        # modify y_true according to the true probability distribution
        if tp == 0: #uniform setting 1
            pn = [1,1]
            pp = 2 * Xt[i]
            prob = (pp[0]*pp[1]*alpha)/(pn[0]*pn[1]*(1-alpha)+pp[0]*pp[1]*alpha)
        elif tp == 1: # tp == 1 it is the Gaussian
            pn = norm.pdf(xnew,loc = 0.5, scale = 0.5)
            pp = norm.pdf(xnew,loc = 1, scale = 0.5)
            prob = (pp[0]*pp[1]*alpha)/(pn[0]*pn[1]*(1-alpha)+pp[0]*pp[1]*alpha)
        elif tp==2:# now it is real data
            pn = [beta.pdf(Xt[i][0], params_1[group]['n'][0], params_1[group]['n'][1]), beta.pdf(Xt[i][1], params_2[group]['n'][0], params_2[group]['n'][1])]
            pp = [beta.pdf(Xt[i][0], params_1[group]['p'][0],params_1[group]['p'][1]), beta.pdf(Xt[i][1], params_2[group]['p'][0],params_2[group]['p'][1])]
            prob = (pp[0]*pp[1]*alpha[group])/(pn[0]*pn[1]*(1-alpha[group])+pp[0]*pp[1]*alpha[group])
        elif tp==4:# now it is real data
            pn = [beta.pdf(Xt[i][0], params_1[group]['n'][0], params_1[group]['n'][1], 0.3,0.3), beta.pdf(Xt[i][1], params_2[group]['n'][0], params_2[group]['n'][1],0.55,0.35)]
            pp = [beta.pdf(Xt[i][0], params_1[group]['p'][0],params_1[group]['p'][1],0.3,0.3), beta.pdf(Xt[i][1], params_2[group]['p'][0],params_2[group]['p'][1],0.55,0.35)]
            prob = (pp[0]*pp[1]*alpha[group])/(pn[0]*pn[1]*(1-alpha[group])+pp[0]*pp[1]*alpha[group])
        else:
            prob = clf.predict_proba(Xt_full[i].reshape(1,-1))[0,1]
        if uniform.rvs() < prob:
            yt_true[i] = 1

    return Xt, yt_true

# Function for simulation
def simulation(Q, N, n, T, alpha, bias='up',mag=0.1,tp = 0,ratio=0.1,loc=0, scale=0.5, params_1={},params_2={}, group = 'a', plot=[], sd = False, refined=False, strategic=True, noise=0, kde_x=None, clf=None, only_human = False):
    """
    The function runs simulation to return Acceptance rates and Qualification rates for T rounds running n times
    Q: cost matrix
    N: The number of agents coming in each round
    n: the total iterations of simulations
    T: total rounds
    alpha: ground truth qualification rate without strategic response
    bias: direction of the systematic bias
    mag: the degree of the bias
    tp: 0 is uniform setting 1, 1 is Gaussian setting 2
    ratio: human-annotated sample ratio
    loc: Gaussian mean
    scale: Gaussian sd
    params_1: beta alpha
    params_2: beta beta
    group: beta distribution group id
    plot: index t we need to output a plot, default is none
    sd: whether calculate sd
    refined: whether using sampler
    strategic: whether stratgeic setting
    noise: the degree of noisy response
    kde_x: semi synthetic data using kde to estimate x
    clf: semi synthetic data ground truth probability
    only_human: only human annotated samples
    """

    # n iterations in total
    # Store means of at and qt

    At_mean = np.zeros(T+1)
    Qt_mean = np.zeros(T+1)

    if sd:
        At_sd = np.zeros(T+1)
        Qt_sd = np.zeros(T+1)
    
    if tp == 3:
        a = 0.1
    else:
        a = 0

    for i in range(n):
        if i%10 == 0:
            print(f'simulation round {i} begins\n')
        
        h_num = int(ratio*N) #human annotated samples every round
    
        # Parameters to store weights, biases, at, qt
        W = dict()
        B = dict()
        Ratio_pred = np.zeros(T+1)
        Ratio_true = np.zeros(T+1)

        # Initial round
        # Generating training data and the incoming data at T = 0
        if tp == 0:
            X_noise_train, y_noise_train = generate_noisy_examples_U(N, alpha, bias, mag)
            X_true_train, y_true_train = generate_true_examples_U(N,alpha)
            X0, y0_true = generate_true_examples_U(N,alpha)
    
        elif tp == 1:
            X_noise_train, y_noise_train = generate_noisy_examples_G(N, alpha, bias, mag, loc, scale)
            X_true_train, y_true_train = generate_true_examples_G(N,alpha,loc,scale)
            X0, y0_true = generate_true_examples_G(N,alpha,loc,scale)
    
        elif tp == 2:
            X_noise_train, y_noise_train = generate_noisy_examples_R(N, alpha,params_1,params_2, bias, mag, group)
            X_true_train, y_true_train = generate_true_examples_R(N,alpha,params_1,params_2,group)
            X0, y0_true = generate_true_examples_R(N,alpha,params_1,params_2,group)
        
        elif tp == 4:
            X_noise_train, y_noise_train = generate_noisy_examples_R2(N, alpha,params_1,params_2, bias, mag, group)
            X_true_train, y_true_train = generate_true_examples_R2(N,alpha,params_1,params_2,group)
            X0, y0_true = generate_true_examples_R2(N,alpha,params_1,params_2,group)         
        
        else:
           X_noise_train, y_noise_train, X_noise_full = generate_noisy_examples_K(N, clf, kde_x, bias, mag)
           X_true_train, y_true_train, X_true_full = generate_true_examples_K(N,clf,kde_x)
           X0, y0_true, X0_full = generate_true_examples_K(N,clf,kde_x)

        # First true classifier
        clf_0_true = get_classifier(X_true_train,y_true_train,a)
        w, b = clf_0_true.coef_, clf_0_true.intercept_

        # First noisy classifier
        clf_0_noise = get_classifier(X_noise_train,y_noise_train,a)

        W[0], B[0] = clf_0_noise.coef_, clf_0_noise.intercept_

        # predict samples X0 with noisy classifier
        y0_pred = clf_0_noise.predict(X0)
        Ratio_pred[0] = len(y0_pred[y0_pred==1])/len(y0_pred) #accpetance rate a0
        Ratio_true[0] = len(y0_true[y0_true==1])/len(y0_true)

        if i == 0 and 0 in plot:
            clf_true = get_classifier(X0, y0_true,a)
            w_true, b_true = clf_true.coef_, clf_true.intercept_
            fig, ax = plt.subplots(1, 1)
            fig.set_size_inches(5,4)
            plot_points(X0, y0_true, ax)
            plot_classifier(w_true, b_true, "true classifier",'green',ax)
            plot_classifier(W[0], B[0], "learned classifier",'black',ax)
            ax.legend()
            fig.savefig(f'illustrate_{bias}_0_{tp}_{ratio}.pdf')

        # Then iterate from 1 to T
        for t in range(1, T+1):
            # get human-annotated data, and the new data coming at t
            if tp == 0:
                X_noise_new, y_noise_new = generate_noisy_examples_U(h_num, alpha, bias, mag)
                Xt, yt_true = generate_true_examples_U(N,alpha)
            elif tp == 1:
                X_noise_new, y_noise_new = generate_noisy_examples_G(h_num, alpha, bias, mag, loc, scale)
                Xt, yt_true = generate_true_examples_G(N,alpha,loc,scale)
            elif tp == 2:
                X_noise_new, y_noise_new = generate_noisy_examples_R(h_num, alpha,params_1,params_2, bias, mag, group)
                Xt, yt_true = generate_true_examples_R(N,alpha,params_1,params_2,group)
            
            elif tp == 4:
                X_noise_new, y_noise_new = generate_noisy_examples_R2(h_num, alpha,params_1,params_2, bias, mag, group)
                Xt, yt_true = generate_true_examples_R2(N,alpha,params_1,params_2,group)

            else:
                # xt full is the full 19 dim array used for generating the ground truth label
                X_noise_new, y_noise_new, X_noise_new_full = generate_noisy_examples_K(h_num,clf,kde_x,bias, mag)
                Xt, yt_true, Xt_full = generate_true_examples_K(N,clf,kde_x)

            # Update training dataset: combination of: (1) previous X_noise_train; (2) Model-annotated X0,y0_pred; (3) X_noise_new

            # only human: no model annotated data
            if only_human:
                X_noise_train = np.concatenate([X_noise_train, X_noise_new])
                y_noise_train = np.concatenate([y_noise_train, y_noise_new])
                if tp==3:
                    X_noise_full = np.concatenate([X_noise_full, X_noise_new_full])
            
            # there is some annotated data
            else:
                # no human data and the dataset is semisynthetic german credit score
                if h_num == 0 and tp==3:
                    X_noise_train = np.concatenate([X_noise_train, X0])
                    y_noise_train = np.concatenate([y_noise_train, y0_pred])
                    X_noise_full = np.concatenate([X_noise_full, X0_full])
                
                # other situations
                else:
                    X_noise_train = np.concatenate([X_noise_train, X0, X_noise_new])
                    y_noise_train = np.concatenate([y_noise_train, y0_pred, y_noise_new])
                    if tp == 3:
                        X_noise_full = np.concatenate([X_noise_full, X0_full, X_noise_new_full])
                        
            # get the updated noisy classifier at t, store coefficients
            clf_noise = get_classifier(X_noise_train,y_noise_train,a)
            W[t], B[t] = clf_noise.coef_, clf_noise.intercept_

            if strategic:
                # previous classifier is parametrized by W[t-1], B[t-1]
                # get the new (feature, label) pairs after best response
                y_previous_pred = np.matmul(Xt, W[t-1].T).reshape(-1) + B[t-1] # predicted values using f_t-1
                y_previous_pred[y_previous_pred >= 0] = 1 # convert value to labels
                y_previous_pred[y_previous_pred < 0] = -1
                if tp == 3:
                    Xt, yt_true = strategic_examples(Xt, yt_true, y_previous_pred, W[t-1], B[t-1], Q, tp=tp, alpha=alpha, loc=loc, scale=scale, params_1 = params_1, params_2 = params_2, group = group, noisy=noise, kde_x=kde_x,clf=clf,Xt_full=Xt_full)
                else:
                    Xt, yt_true = strategic_examples(Xt, yt_true, y_previous_pred, W[t-1], B[t-1], Q, tp=tp, alpha=alpha, loc=loc, scale=scale, params_1 = params_1, params_2 = params_2, group = group, noisy=noise, kde_x=kde_x,clf=clf)

            # Get qt (true quailification rate) at time t
            Ratio_true[t] = len(yt_true[yt_true==1])/len(yt_true)

            # Get at (acceptance rate) at time t
            yt_pred = clf_noise.predict(Xt)
            Ratio_pred[t] = len(yt_pred[yt_pred==1])/len(yt_pred)

            # if need to output a plot
            if i == 0 and t in plot:
                clf_true = get_classifier(Xt, yt_true,a)
                w_true, b_true = clf_true.coef_, clf_true.intercept_
                fig, ax = plt.subplots(1, 1)
                fig.set_size_inches(5,4)
                plot_points(Xt, yt_true, ax)
                plot_classifier(w_true, b_true, "true classifier",'green',ax)
                plot_classifier(W[t], B[t], "learned classifier",'black',ax)
                ax.legend()
                fig.savefig(f'illustrate_{bias}_{t}_{tp}_{ratio}.pdf')
            
            # Then update X0 to model-annotated samples at t
            if refined:
                yt_prob = clf_noise.predict_proba(Xt)[:,1]
                X0 = Xt
                rad = np.array(uniform.rvs(size = len(yt_prob)))
                rad[rad < yt_prob] = 1
                # the label is different in German credit data
                if tp == 3:
                    rad[rad < 1] = 0
                else:
                    rad[rad < 1] = -1
                y0_pred = rad
                y0_true = yt_true

            # normally, no sampler is used
            else:
                X0 = Xt
                y0_pred = yt_pred
                y0_true = yt_true

        
        # Save the mean values of at and qt
        At_mean += 1/n*Ratio_pred
        Qt_mean += 1/n*Ratio_true

        if sd:
            At_sd += 1/n*Ratio_pred**2
            Qt_sd += 1/n*Ratio_true**2
    # save standard deviation
    if sd:
        At_sd = np.sqrt(At_sd - At_mean**2)
        Qt_sd = np.sqrt(Qt_sd - Qt_mean**2)
        At_sd = np.nan_to_num(At_sd)
        Qt_sd = np.nan_to_num(Qt_sd)

        return At_mean, Qt_mean, At_sd, Qt_sd

    return At_mean, Qt_mean

# Function to plot At, Qt and save the results
def plot_save_single(At_mean, Qt_mean, des, save = False, limit = False):
    # Plot at, qt, deltat of a single group
    # then save the 3 average results into text files
    import matplotlib as mpl
    mpl.rcParams['lines.markersize'] = 6
    Deltat_mean = abs(At_mean - Qt_mean)
    if save:
        np.savetxt(f'results_text/{des}.out', (At_mean,Qt_mean,Deltat_mean), delimiter=',')
    plt.figure(figsize = (4,3))
    if limit:
        x = list(range(0, len(At_mean), 3))
        plt.plot(x, At_mean[x], marker = 'o', color = 'red',markersize=5, label = r'$a_t$')
        plt.plot(x, Qt_mean[x], marker = 's', color = 'blue', markersize=5,label = r'$q_t$')
        plt.plot(x, Deltat_mean[x], marker = 'v', color = 'black', markersize=5,label = r'$\Delta_t$')
        plt.xticks(range(0,len(At_mean), 8),fontsize = 12)
    else:
        x = list(range(len(At_mean)))
        plt.plot(x, At_mean, marker = 'o', color = 'red', label = r'$a_t$')
        plt.plot(x, Qt_mean, marker = 's', color = 'blue', label = r'$q_t$')
        plt.plot(x, Deltat_mean, marker = 'v', color = 'black', label = r'$\Delta_t$')
        plt.xticks(range(0,len(At_mean)+1, 2),fontsize = 14)
    
    plt.xlabel(r'$t$', fontsize = 14)
    plt.yticks(np.arange(0, 1.01, step=0.2),fontsize = 14)
    plt.legend(labelspacing=0.5,handlelength=1)
    if save:
        plt.savefig(f'plots_new/{des}.pdf', bbox_inches='tight')
    return At_mean, Qt_mean, Deltat_mean


# Function with error bar
def plot_save_single_err(At_mean, Qt_mean, At_sd, Qt_sd, des, save = False, limit=False, small=False):
    # Plot at, qt, deltat of a single group
    # then save the 3 average results into text files
    import matplotlib as mpl
    mpl.rcParams['lines.markersize'] = 6
    if small:
        mpl.rcParams['lines.markersize'] = 1
    Deltat_mean = abs(At_mean - Qt_mean)
    Deltat_sd = np.sqrt(At_sd**2 + Qt_sd**2)
    if save:
        np.savetxt(f'results_text/{des}.out', (At_mean,Qt_mean,Deltat_mean), delimiter=',')
        np.savetxt(f'results_text/ebar_{des}.out', (At_sd, Qt_sd, Deltat_sd), delimiter=',')
    plt.figure(figsize = (4,3))
    if limit:
        x = list(range(0, len(At_mean), 3))
        plt.errorbar(x, At_mean[x], yerr=At_sd[x], marker = 'o', color = 'red', markersize=5,label = r'$a_t$')
        plt.errorbar(x, Qt_mean[x], yerr=Qt_sd[x], marker = 's', color = 'blue', markersize=5, label = r'$q_t$')
        plt.errorbar(x, Deltat_mean[x], yerr=Deltat_sd[x], marker = 'v', color = 'black',markersize=5, label = r'$\Delta_t$')
        plt.xticks(range(0,len(At_mean), 8),fontsize = 12)
    else:
        x = list(range(len(At_mean)))
        plt.errorbar(x, At_mean, yerr=At_sd, marker = 'o', color = 'red', label = r'$a_t$')
        plt.errorbar(x, Qt_mean, yerr=Qt_sd, marker = 's', color = 'blue', label = r'$q_t$')
        plt.errorbar(x, Deltat_mean, yerr=Deltat_sd, marker = 'v', color = 'black', label = r'$\Delta_t$')
        plt.xticks(range(0,len(At_mean)+1, 2),fontsize = 14)
    
    plt.xlabel(r'$t$', fontsize = 14)
    plt.yticks(np.arange(0, 1.01, step=0.2),fontsize = 14)
    plt.legend(labelspacing=0.5,handlelength=1)
    if save:
        plt.savefig(f'plots_new/ebar_{des}.pdf', bbox_inches='tight')
    return At_mean, Qt_mean, Deltat_mean, At_sd, Qt_sd, Deltat_sd


# Function to plot fairness (2 groups participate) and save the two results
def plot_save_fairness(Ati_mean, Atj_mean, Qti_mean, Qtj_mean, des, save=False):
    import matplotlib as mpl
    mpl.rcParams['lines.markersize'] = 6
    Deltati_mean = abs(Ati_mean - Qti_mean)
    Deltatj_mean = abs(Atj_mean - Qtj_mean)
    if save:
        np.savetxt(f'results_text/{des}.out', (Ati_mean, Atj_mean, Ati_mean-Atj_mean),delimiter=',')
    # plot the DP fairness acceptance rate
    plt.figure(figsize = (4,3))
    x = list(range(len(Ati_mean)))
    plt.plot(x, Ati_mean, marker = 'o', color = 'red', label = r'$a_t$ for i')
    plt.plot(x, Atj_mean, marker = 's', color = 'blue', label = r'$a_t$ for j')
    plt.plot(x, abs(Ati_mean - Atj_mean), marker = 'v', color = 'black', label = r'Unfairness')
    plt.xticks(range(0,len(Ati_mean)+1, 2),fontsize=14)
    plt.yticks(np.arange(0, 1.01, step=0.2),fontsize=14)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.legend(labelspacing=0.5,handlelength=1,fontsize=14,fancybox=True, framealpha=0.5)
    if save:
        plt.savefig(f'plots_new/fair_{des}.pdf', bbox_inches='tight')
    # Plot the classifier bias of the two groups
    plt.figure(figsize = (4,3))
    plt.plot(x, Deltati_mean, marker = 'o', color = 'red', label = r'$\Delta_t$ for i')
    plt.plot(x, Deltatj_mean, marker = 's', color = 'blue', label = r'$\Delta_t$ for j')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.xticks(range(0,len(Ati_mean)+1, 2),fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(labelspacing=0.5,handlelength=1,fontsize=14,fancybox=True, framealpha=0.5)
    if save:
        plt.savefig(f'plots_new/bias_{des}.pdf',bbox_inches='tight')
    return Ati_mean, Qti_mean, Deltati_mean, Atj_mean, Qtj_mean, Deltatj_mean



# Function to plot fairness (2 groups participate) and save the two results with err bars
def plot_save_fairness_err(Ati_mean, Atj_mean, Qti_mean, Qtj_mean, Ati_sd, Qti_sd, Atj_sd, Qtj_sd, des, save=False, small=False):
    import matplotlib as mpl
    mpl.rcParams['lines.markersize'] = 6
    if small:
        import matplotlib as mpl
        mpl.rcParams['lines.markersize'] = 1
    Deltati_mean = abs(Ati_mean - Qti_mean)
    Deltatj_mean = abs(Atj_mean - Qtj_mean)
    Deltati_sd = np.maximum(Ati_sd,Qti_sd)
    Deltatj_sd = np.maximum(Atj_sd,Qtj_sd)
    unfair_sd = np.maximum(Ati_sd,Atj_sd)
    if save:
        np.savetxt(f'results_text/{des}.out', (Ati_mean, Atj_mean, Ati_mean-Atj_mean),delimiter=',')
        np.savetxt(f'results_text/ebar_{des}.out', (Ati_sd, Atj_sd, unfair_sd), delimiter=',')
    # plot the DP fairness acceptance rate
    plt.figure(figsize = (4,3))
    x = list(range(len(Ati_mean)))
    plt.errorbar(x, Ati_mean, yerr=Ati_sd, marker = 'o', color = 'red', label = r'$a_t$ for i')
    plt.errorbar(x, Atj_mean, yerr=Atj_sd, marker = 's', color = 'blue', label = r'$a_t$ for j')
    plt.errorbar(x, abs(Ati_mean - Atj_mean), yerr=unfair_sd, marker = 'v', color = 'black', label = r'Unfairness')
    plt.xticks(range(0,len(Ati_mean)+1, 2),fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.legend(labelspacing=0.5,handlelength=1,fontsize=14,fancybox=True, framealpha=0.5)
    if save:
        plt.savefig(f'plots_new/ebar_fair_{des}.pdf', bbox_inches='tight')
    # Plot the classifier bias of the two groups
    plt.figure(figsize = (4,3))
    plt.errorbar(x, Deltati_mean, yerr=Deltati_sd, marker = 'o', color = 'red', label = r'$\Delta_t$ for i')
    plt.errorbar(x, Deltatj_mean, yerr=Deltatj_sd, marker = 's', color = 'blue', label = r'$\Delta_t$ for j')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.xticks(range(0,len(Ati_mean)+1, 2),fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(labelspacing=0.5,handlelength=1,fontsize=14,fancybox=True, framealpha=0.5)
    if save:
        plt.savefig(f'plots_new/ebar_bias_{des}.pdf',bbox_inches='tight')
    mpl.rcParams['lines.markersize'] = 10
    return Ati_mean, Qti_mean, Deltati_mean, Atj_mean, Qtj_mean, Deltatj_mean

    

# Function to read At_mean, Qt_mean, At_sd, Qt_sd
def read_results(des):
    with open(f'results_text/{des}.out') as f:
        lines = f.readlines()
        At_mean = np.array([float(i) for i in lines[0].split(",") if i.strip()])
        Qt_mean = np.array([float(i) for i in lines[1].split(",") if i.strip()])
    
    with open(f'results_text/ebar_{des}.out') as f:
        lines = f.readlines()
        At_sd = np.array([float(i) for i in lines[0].split(",") if i.strip()])
        Qt_sd = np.array([float(i) for i in lines[1].split(",") if i.strip()])
    
    return At_mean, Qt_mean, At_sd, Qt_sd