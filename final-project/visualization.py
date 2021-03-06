def learning_curve_plot(loss_list, mae_list,
                        title='Learning Curve of \n Gaussian Model'):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()

    color = 'RoyalBlue'
    ax1.set_title(title)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Training Loss (ELBO)', color=color)
    ax1.plot(range(len(loss_list)), loss_list, marker='o',
             label='Training Loss (ELBO)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'orange'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Test Loss (MAE)', color=color)
    ax2.plot(range(len(mae_list)), mae_list, marker='v',
             label='Test Loss (MAE)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend()
    plt.grid()
    plt.show()


def latent_viz(u="u_mean", v="v_mean",
               title="U and V Visualization of Normal Model SVI Method"):
    import pyro
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(title)
    ax1.imshow(pyro.param("u_mean").t().detach().numpy(),
               interpolation='nearest')
    ax1.set_title("U Transpose")
    ax2.imshow(pyro.param("v_mean").t().detach().numpy(),
               interpolation='nearest')
    ax2.set_title("V Transpose")
    fig.subplots_adjust(right=0.8)
    fig.colorbar(ax1.imshow(pyro.param("u_mean").t().detach().numpy(),
                            interpolation='nearest'),
                 orientation="horizontal")


def hist_comparison(true_ratings, pred_ratings, title="Gaussian Model"):
    import matplotlib.pyplot as plt
    plt.hist(true_ratings, bins=20, alpha=0.7, label="Real Ratings")
    plt.hist(pred_ratings, bins=20, alpha=0.7, label="Predicted Ratings")
    plt.legend(prop={'size': 16})
    plt.title(title)
    plt.xlabel('Rating')
    plt.show()
