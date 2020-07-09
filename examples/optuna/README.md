# Example of Optuna-powered Hyper-parameters Tuning

Tune hyper parameters using [Optuna](https://optuna.org/).

Although the example script (`optuna_dqn_lunarlander.py`) uses fixed target algorithm/environment
in order to fucus on the concept of Optuna-powered PFRL,
you can create one for your own use easily thanks to the Optuna's high flexibility!


## How to Run

### Quickstart

The quickstart on your local machine.

```bash
storage="sqlite:///example.db"
study="optuna-pfrl-quickstart"

# In common RL, higher score means better performance (`--direction maximize`)
optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

# Start tuning hyper parameters
python optuna_dqn_lunarlander.py --optuna-study-name "${study}" --optuna-storage "${storage}"
```

You can see the dashboard (experimental):

```bash
pip install bokeh==1.4.0  # https://github.com/optuna/optuna/issues/1320
optuna dashboard --study-name "${study}" --storage "${storage}"
```

Then open `localhost:5006` on your browser.

![dashboard](assets/dashboard.png)



### Distributed Optimization

You might have already noticed that the sample script can be executed in parallel, and distributed:

```bash
# DB specs. We assume PostgreSQL here but you can use various backend DB engines.
# Note that SQLite is not recommended for parallel optimization.
postgres_user="user"
postgres_password="password"
postgres_host="host"
postgres_database"database"

storage="postgresql://${postgres_user}:${postgres_password}@${postgres_host}/${postgres_database}"
study="optuna-pfrl-distributed"

optuna create-study --study-name "${study}" --storage "${storage}" --direction maximize

# You can run two processes parallelly (If your computation resource allows!)
python optuna_dqn_lunarlander.py --optuna-study-name "${study}" --optuna-storage "${storage}" &
python optuna_dqn_lunarlander.py --optuna-study-name "${study}" --optuna-storage "${storage}" &
```

```bash
# And Optuna works wherever the backend DB is accessible
ssh some-server

postgres_user="user"
postgres_password="password"
postgres_host="host"
postgres_database"database"

storage="postgresql://${postgres_user}:${postgres_password}@${postgres_host}/${postgres_database}"
study="optuna-pfrl-distributed"

python optuna_dqn_lunarlander.py --optuna-study-name "${study}" --optuna-storage "${storage}"
```
