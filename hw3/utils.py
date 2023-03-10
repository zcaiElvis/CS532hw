import wandb

def wandb_plots_homework3(samples, program):

    # W&B logging of actual plots
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='mu', title='Program 1; mu')
    elif program == 2:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'slope', 'bias'])
        wandb_log['Program 2; slope'] = wandb.plot.histogram(table, value='slope', title='Program 2; slope')
        wandb_log['Program 2; bias'] = wandb.plot.histogram(table, value='bias', title='Program 2; bias')
        wandb_log['Program 2; scatter'] = wandb.plot.scatter(table, x='slope', y='bias', title='Program 2; slope vs. bias')
    elif program == 3:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x'])
        wandb_log['Program 3'] = wandb.plot.histogram(table, value='x', title='Program 3; Are the points from the same cluster?')
    elif program == 4:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x'])
        wandb_log['Program 4'] = wandb.plot.histogram(table, value='x', title='Program 4; Is it raining?')
    elif program == 5:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x', 'y'])
        wandb_log['Program 5; x'] = wandb.plot.histogram(table, value='x', title='Program 5; x')
        wandb_log['Program 5; y'] = wandb.plot.histogram(table, value='y', title='Program 5; y')
        wandb_log['Program 5; scatter'] = wandb.plot.scatter(table, x='x', y='y', title='Program 5; x vs. y')
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)