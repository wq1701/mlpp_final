# Gaussian Prior


def run_gaussian_mf(anime_matrix_train, anime_data_test, k,
                    method="svi", lr=0.05, n_steps=1000, mae_tol=0.02):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import torch
    from torch.distributions import constraints
    from torch.autograd import Variable

    import pyro
    import pyro.distributions as dist
    import pyro.optim as optim

    from pyro.infer import SVI, Trace_ELBO

    from pyro.infer.mcmc.api import MCMC
    from pyro.infer.mcmc import NUTS

    # from preprocess import fill_unrated, control_value
    from eval_metrics import mae

    # from sklearn.model_selection import train_test_split

    pyro.set_rng_seed(1)
    assert pyro.__version__.startswith('1.4.0')

    sigma_u = torch.tensor(1.0)
    sigma_v = torch.tensor(1.0)

    def matrix_factorization_normal(anime_matrix_train, k=k):

        m = anime_matrix_train.shape[0]
        n = anime_matrix_train.shape[1]

        u_mean = Variable(torch.zeros([m, k]))
        u_sigma = Variable(torch.ones([m, k]) * sigma_u)

        v_mean = Variable(torch.zeros([n, k]))
        v_sigma = Variable(torch.ones([n, k]) * sigma_v)

        u = pyro.sample("u", dist.Normal(
            loc=u_mean, scale=u_sigma).to_event(2))
        v = pyro.sample("v", dist.Normal(
            loc=v_mean, scale=v_sigma).to_event(2))

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
                                obs=valid_matrix)

    def guide_map(anime_matrix_train, k=k):
        m = anime_matrix_train.shape[0]
        n = anime_matrix_train.shape[1]

        u_map = pyro.param('u_map', torch.zeros([m, k]))
        v_map = pyro.param('v_map', torch.zeros([n, k]))
        sigma_map = pyro.param("sigma_map", torch.tensor(
            1.0), constraint=constraints.positive)

        pyro.sample("u", dist.Delta(u_map).to_event(2))
        pyro.sample("v", dist.Delta(v_map).to_event(2))
        pyro.sample("sigma", dist.Delta(sigma_map))

    def guide_svi(anime_matrix_train, k=k):
        m = anime_matrix_train.shape[0]
        n = anime_matrix_train.shape[1]

        u_mean = pyro.param('u_mean', torch.zeros([m, k]))
        u_sigma = pyro.param('u_sigma', torch.ones(
            [m, k]) * sigma_u, constraint=constraints.positive)

        v_mean = pyro.param('v_mean', torch.zeros([n, k]))
        v_sigma = pyro.param('v_sigma', torch.ones(
            [n, k]) * sigma_v, constraint=constraints.positive)

        sigma_loc = pyro.param('sigma_loc', torch.tensor(1.),
                               constraint=constraints.positive)
        sigma_scale = pyro.param('sigma_scale', torch.tensor(0.1),
                                 constraint=constraints.positive)

        pyro.sample("u", dist.Normal(u_mean, u_sigma).to_event(2))
        pyro.sample("v", dist.Normal(v_mean, v_sigma).to_event(2))
        pyro.sample("sigma", dist.Normal(sigma_loc, sigma_scale))

    def train_via_opt_map(model, guide):
        pyro.clear_param_store()
        svi = SVI(model, guide, optim.Adam({"lr": lr}), loss=Trace_ELBO())

        loss_list = []
        mae_list = []
        for step in range(n_steps):
            loss = svi.step(anime_matrix_train.values)
            pred = []
            for i, j in anime_data_test[
                    ["user_id", "anime_id"]].itertuples(index=False):
                r = torch.dot(pyro.param("u_map")[
                              i - 1, :], pyro.param("v_map")[j - 1, :])
                r = r.item()
                if r > 10:
                    r = 10
                pred.append(r)
            test_mae = mae(anime_data_test.rating, pred)
            if step > 500 and test_mae - min(mae_list) > mae_tol:
                print('[stop at iter {}] loss: {:.4f} Test MAE: {:.4f}'.format(
                    step, loss, test_mae)
                )
                break
            loss_list.append(loss)
            mae_list.append(test_mae)
            if step % 250 == 0:
                print('[iter {}]  loss: {:.4f} Test MAE: {:.4f}'.format(
                    step, loss, test_mae))
        return(loss_list, mae_list)

    def train_via_opt_svi(model, guide):
        pyro.clear_param_store()
        svi = SVI(model, guide, optim.Adam({"lr": lr}), loss=Trace_ELBO())

        loss_list = []
        mae_list = []
        for step in range(n_steps):
            loss = svi.step(anime_matrix_train.values)
            pred = []
            for i, j in anime_data_test[["user_id",
                                         "anime_id"]].itertuples(index=False):
                r = torch.dot(pyro.param("u_mean")[
                              i - 1, :], pyro.param("v_mean")[j - 1, :])
                r = r.item()
                if r > 10:
                    r = 10
                pred.append(r)
            test_mae = mae(anime_data_test.rating, pred)
            if step > 500 and test_mae - min(mae_list) > mae_tol:
                print('[stop at iter {}] loss: {:.4f} Test MAE: {:.4f}'.format(
                    step, loss, test_mae)
                )
                break
            loss_list.append(loss)
            mae_list.append(test_mae)
            if step % 250 == 0:
                print('[iter {}]  loss: {:.4f} Test MAE: {:.4f}'.format(
                    step, loss, test_mae))
        return(loss_list, mae_list)

    def hmc(matrix_factorization_normal, num_samples=1000, warmup_steps=200):
        nuts_kernel = NUTS(matrix_factorization_normal)
        mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
        mcmc.run(anime_matrix_train.values, k=k)
        hmc_samples = {k: v.detach().cpu().numpy()
                       for k, v in mcmc.get_samples().items()}
        return hmc_samples

    if method == "map":
        loss_list, mae_list = train_via_opt_map(
            matrix_factorization_normal, guide_map)
    if method == "svi":
        loss_list, mae_list = train_via_opt_svi(
            matrix_factorization_normal, guide_svi)
    if method == "hmc":
        return hmc(matrix_factorization_normal, num_samples=1000,
                   warmup_steps=200)

    return loss_list, mae_list
