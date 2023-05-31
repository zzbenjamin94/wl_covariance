                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 estimator
def chisqr(model, obs, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-model[i])**2)/(error[i]**2)
    return chisqr


'''
Prints the GelmanRubin R coefficient. R < 1.1 for convergence among walkers. 

chain: UNFLATTENED posterior chain of dimensions (Nsteps, Nwalkers, Nparameter)
burn: (Optional) for burn in. Typically not needed as burn in done before hand in posterior_summary_stats()
maxlag: (Optional) the maximal autocorrelation computed before function terminates
'''

def check_chains(chain, burn=0, maxlag=1000):
    '''
    Ignoring `burn` samples from the front of each chain, compute convergence criteria and
    effective number of samples.
    '''
    nsteps, nwalk, npars = chain.shape
    if burn >= nsteps: return
    tmp_samples = [chain[burn:,i,:] for i in range(nwalk)]
    print('R =', cr.GelmanRubinR(tmp_samples))
    #print('neff =', cr.effective_samples(tmp_samples, maxlag=maxlag))
    #print('NB: Since walkers are not independent, these will be optimistic!')
    return


'''
chain: np.array in dimensions of (nsteps, nwalk, npars).  
model: The functional form of the model
Ndim: number of parameters for the model
x_data: the x data vector
y_data: the y data vector
y_err: error on the y data vector
verbose: If true, displays the best fit parameters and bounds, DIC, BIC and Chi-square right-tail p-value

Returns:
best_fit_params: (Ndim,3) array of the best fit parameters. For each parameter the 2nd axes denote [best_fit_val, upper_err, lower_err] using the (16,50,84) percentile bounds 
DIC_model: The DIC of the model
BIC_model: The BIC of the model
p_val: Chi-square right tail p-value
'''
def posterior_summary_stats(chain, model, Ndim, x_data, y_data, y_err, verbose=False):
 
    
    ##Turn into a flatchain
    flatchain = chain.reshape((-1,Ndim)).T
    
    ##Best fit parameters
    params_iter = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(flatchain, [16, 50, 84], axis=1)))
        
    best_fit_params = []
    for i, param_cur in enumerate(params_iter):
        best_fit_params.append(list(param_cur))
        
    best_fit_params = np.array(best_fit_params)
    model_best_fit = model(best_fit_params[:,0], x_data)
    
    ##Chi-square p-value
    p_val = 1 - stats.chi2.cdf(chisqr(model_best_fit, y_data, y_err) , len(y_data)-Ndim)
    
    ###DIC 
    reduce_sample = 1 #int(np.shape(flatchain)[1]/100) ## Count every 1000  
    DIC_model =  DIC(model, flatchain[:,::reduce_sample],  x_data, y_data, y_err)
    BIC_model =  BIC(model, flatchain[:,::reduce_sample],  x_data, y_data, y_err)
    
    if verbose:
        ##Check the convergence of chains
        check_chains(chain)
        
        ##Best fit params
        for i in range(len(best_fit_params)):
            print('param_{}: best fit, upper_err, lower_err: '.format(i), best_fit_params[i])
            
        ##DIC and BIC
        print("DIC: ", DIC_model)
        print("BIC: ", BIC_model)

        ##Chi-square p-value
        print('Chi-square p-value: ', p_val)
        
    return best_fit_params, DIC_model, BIC_model, p_val



'''
Runs the chains, outputs and prints summary stats, and saves the chains onto a H5PY file. 

Nwalkers: Number of walkers
Ndim: Number of dimensions for the model
Nsample: Number of sample per chain
x_data: x data vector
y_data: y data vector
y_err: error on the y data vector
x0: Initialization parameters for the model, should be length Ndim
burn_in: (Optional) number of posterior samples to burn
thin: Thin by amount
maxlag: Max autocorrelation lag to compute to check chain convergence
savefile: Boolean to save the file
filename: directory and filename
verbose: If true, outputs Starting params minimization convergence, autocorrelation time as computed by GW10, GelmanRubin R correlation, DIC, BIC, Chi-square left tail p-value

Returns:
best_fit_params: (Ndim,3) array of the best fit parameters. For each parameter the 2nd axes denote [best_fit_val, upper_err, lower_err] using the (16,50,84) percentile bounds 
chain: UNFLATTENED chain of dimension (Nsteps, Ndim, Nparam) after thinning and burn in

'''
def run_model_chains(Nwalkers, Ndim, Nsample, model, x_data, y_data, y_err, x0,
                     burn_in=500, thin=100, maxlag=1000, savefile=True, filename='test.h5py', verbose=False): 
        
    ##Initial guess
    ## use the scipy minimze method to find the best fit parameters as the starting point
    nll = lambda *args: -lnlike(*args)
    result = opt.minimize(nll, x0=x0,
                              args=(model, x_data, y_data, y_err), method='Nelder-Mead', options={'gtol': 1e-6, 'disp': verbose}) 
    starting_params = result['x']
    
    p0 = np.array([starting_params*(1.0 + 0.01*np.random.randn(Ndim)) for j in range(Nwalkers)])


    # Initialize the sampler        
    sampler = emcee.EnsembleSampler(Nwalkers, Ndim, lnprob,
                    args=(model, x_data, y_data, y_err)) ##No backend
        
    sampler.run_mcmc(p0, Nsample, progress=verbose) 
    chain = sampler.get_chain(flat=False, thin=thin, discard=burn_in)
    flatchain = sampler.get_chain(flat=True, thin=thin, discard=burn_in).T
    best_fit_params, DIC_model, BIC_model, p_val = posterior_summary_stats(chain, model, Ndim, x_data, y_data, y_err, verbose=verbose)
    
    if verbose:
        print("Starting params: ", starting_params, "Minimization convergence: ", result['success'])
        tau = sampler.get_autocorr_time(quiet=True)
        #print("Autocorrelation: ", tau)
        #check_chains(chain, 1, maxlag=maxlag)
        
    
    sampler.reset()
 
    ##Save the files
    if savefile:
        with h5py.File(filename, "w") as f:
                    f.create_dataset("chains",data=chain)

    return best_fit_params, chain