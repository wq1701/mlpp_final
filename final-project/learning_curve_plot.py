def learning_curve_plot(loss_list, mae_list, title = 'Learning Curve of \n Normal Model'):
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots()
    
    color = 'RoyalBlue'
    ax1.set_title(title)
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Training Loss (ELBO)', color = color)
    ax1.plot(range(len(loss_list)), loss_list, marker='o', label='Training Loss (ELBO)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'orange'
    ax2.set_ylabel('Test Loss (MAE)', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(len(mae_list)), mae_list, marker='v', label='Test Loss (MAE)', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend()
    plt.grid()
    plt.show()


