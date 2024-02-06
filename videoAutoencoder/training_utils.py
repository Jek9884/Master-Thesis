from ray import air
from ray import tune

def grid_search(train_data, val_data, params, metric, device):

    params['train_data'] = train_data
    params['val_data'] = val_data

    params['metric'] = metric
    params['device'] = device

    # Set logs to be shown on the Command Line Interface every 30 seconds
    reporter = tune.CLIReporter(max_report_frequency=30)

    # Starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                       run_config=air.RunConfig(name=params['optimizer'], verbose=1, progress_reporter=reporter))
    results = tuner.fit()

    # Get a dataframe for the last reported results of all of the trials 
    df = results.get_dataframe()


def trainable(config_dict):
    raise NotImplementedError
