from lightautoml.dataset.roles import DatetimeRole
from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML


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

    return automl, oof_pred