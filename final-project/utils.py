#%% Normal dist. prior
def matrix_factorization_normal(anime_matrix_train, k = k):
    
    m = anime_matrix_train.shape[0]
    n = anime_matrix_train.shape[1]
    
    u_mean = Variable(torch.zeros([m, k]))
    u_sigma = Variable(torch.ones([m, k]) * sigma_u)

    v_mean = Variable(torch.zeros([n, k]))
    v_sigma = Variable(torch.ones([n, k]) * sigma_v)

    u = pyro.sample("u", dist.Normal(loc = u_mean, scale = u_sigma).to_event(2))
    v = pyro.sample("v", dist.Normal(loc = v_mean, scale = v_sigma).to_event(2))
    
    expectation = torch.mm(u, v.t())
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
    is_observed = (~np.isnan(anime_matrix_train))
    is_observed = torch.tensor(is_observed)
    valid_matrix = torch.tensor(anime_matrix_train).clone()
    valid_matrix[~is_observed] = 0  # ensure all values are valid
    
    with pyro.plate("user", m, dim=-2): 
        with pyro.plate("anime", n, dim=-3):
            with pyro.poutine.mask(mask=is_observed):
                pyro.sample("obs", dist.Normal(expectation, sigma), 
                        obs = valid_matrix)

# Utility function to print latent sites' quantile information.
def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

#%% MAP guide
def guide_map(anime_matrix_train, k = k):
    m = anime_matrix_train.shape[0]
    n = anime_matrix_train.shape[1]
    
    u_map = pyro.param('u_map', torch.zeros([m, k]))
    v_map = pyro.param('v_map', torch.zeros([n, k]))
    sigma_map = pyro.param("sigma_map", torch.tensor(1.0), constraint=constraints.positive)
    
    pyro.sample("u", dist.Delta(u_map).to_event(2))
    pyro.sample("v", dist.Delta(v_map).to_event(2))
    pyro.sample("sigma", dist.Delta(sigma_map))

#%% MAP inference
def train_via_opt(model, guide, lr=0.05, n_steps = 2000, mae_tol = 0.1):
    pyro.clear_param_store()
    svi = SVI(model, guide, optim.Adam({"lr": lr}), loss=Trace_ELBO())
    
    loss_list = []
    mae_list = []
    for step in range(n_steps):
        loss = svi.step(anime_matrix_train.values)
        pred = []
        for i,j in anime_data_test[["user_id", "anime_id"]].itertuples(index=False):
            r = torch.dot(pyro.param("u_map")[i-1, :], pyro.param("v_map")[j-1, :])
            r = r.item()
            pred.append(r)
        MAE = mae(anime_data_test.rating, pred)
        if step > 500 and MAE - min(mae_list) > mae_tol:
            print('[stop at iter {}]  loss: {:.4f} Test MAE: {:.4f}'.format(step, loss, MAE))
            break
        loss_list.append(loss)
        mae_list.append(MAE)
        if step % 250 == 0:
            print('[iter {}]  loss: {:.4f} Test MAE: {:.4f}'.format(step, loss, MAE))
    return(loss_list, mae_list)

