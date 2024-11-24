from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from sklearn.metrics import mean_absolute_error as mae

def modeling(df):
    random_state = 42
    target = 'SalePrice'
    task = Task('reg', metric='mae')
    roles = {
        'target': target,
    }
    timeout = 3600
    threads = 4
    cv = 5

    automl = TabularAutoML(
        task=task,
        timeout=timeout,
        cpu_limit=threads,
        reader_params={'n_jobs': threads, 'random_state': random_state, 'cv': cv}
    )

    oof_pred = automl.fit_predict(df, roles=roles, verbose=2)
    score = mae(df['SalePrice'], oof_pred.data.ravel())
    print(f'MAE score: {score}')

    automl_utilized = TabularUtilizedAutoML(
        task=task,
        timeout=timeout,
        cpu_limit=threads,
        reader_params={'n_jobs': threads, 'random_state': random_state, 'cv': cv}
    )

    oof_pred_utilized = automl_utilized.fit_predict(df, roles=roles, verbose=2)
    score = mae(df['SalePrice'], oof_pred_utilized.data.ravel())
    print(f'MAE score: {score}')

    return automl, oof_pred, automl_utilized, oof_pred_utilized