# Hyperparameter Tuning with Optuna

Tune hyper parameters using [Optuna](https://optuna.org/).

Although the example script (`optuna_dqn_obs1d.py`) uses fixed target algorithm/environment
(i.e., DQN and environments with 1d continuous observation space and discrete action space)
in order to fucus on the concept of Optuna-powered PFRL,
you can create one for your own use easily thanks to the Optuna's high flexibility!


## How to Run

### Quickstart

The quickstart on your local machine.

```bash
storage="sqlite:///example.db"
study="optuna-pfrl-quickstart"
pruner="HyperbandPruner"

# In RL, higher score means better performance ("--direction maximize")
optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

# Start tuning hyper parameters
python optuna_dqn_obs1d.py --optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-pruner "${pruner}"
```

You can see the optimization history of this study on [Jupyter Notebook](https://jupyter.org/install)
via the [Optuna visualization module](https://optuna.readthedocs.io/en/latest/reference/visualization.html):

```python
# On jupyter notebook
import optuna

study_name = "optuna-pfrl-quickstart"
storage = "sqlite:///example.db"
study = optuna.load_study(study_name=study_name, storage=storage)

optuna.visualization.plot_optimization_history(study)
```

![optimization_history](assets/optimization_history.png)


If you are interested in the `--optuna-pruner` argument above, see the
[corresponding Optuna document](https://optuna.readthedocs.io/en/latest/reference/pruners.html).


### Distributed Optimization

The script can also be used as a "worker process" for parallel and distributed executions.

#### Prerequisites

Since [SQLite is not recommended for parallel optimization](https://optuna.readthedocs.io/en/latest/tutorial/004_distributed.html#distributed-optimization),
we'll use PostgreSQL instead of SQLite hereafter.  

- **Can access PostgreSQL** database named `${postgres_database}` running on a server `${host}`.
- `psycopg2` (PostgreSQL python wrapper) https://pypi.org/project/psycopg2/
  - `pip install psycopg2-binary` for installing stand-alone package
  - `pip install psycopg2` also works when you [have PostgreSQL libraries on your machine](https://www.psycopg.org/docs/install.html#prerequisites).

#### Create a study

```bash
# DB specs. We assume PostgreSQL here but you can use various backend DB engines.
# Here, you must be able to access to a PostgreSQL database named ${postgres_database}
# running on a server ${host}.
postgres_user="user"
postgres_password="password"
postgres_host="host"
postgres_database="database"

storage="postgresql://${postgres_user}:${postgres_password}@${postgres_host}/${postgres_database}"
study="optuna-pfrl-distributed"
pruner="HyperbandPruner"

optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize
```

#### Run the optimization

The script bellow can work wherever the backend DB is accessible.
For distributed optimization, just run this script on multiple servers.

```bash
postgres_user="user"
postgres_password="password"
postgres_host="host"
postgres_database="database"

storage="postgresql://${postgres_user}:${postgres_password}@${postgres_host}/${postgres_database}"
study="optuna-pfrl-distributed"
pruner="HyperbandPruner"

# You can run two processes parallelly (If computation resource allows)
python optuna_dqn_obs1d.py --optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-pruner "${pruner}" &
python optuna_dqn_obs1d.py --optuna-study-name "${study}" --optuna-storage "${storage}" --optuna-pruner "${pruner}" &
```
