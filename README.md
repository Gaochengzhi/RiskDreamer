# RiskDreamer


## Before running 
### 1. conda env setup:
```shell
conda create dr3 -n python=3.11
conda activate dr3
cd dreamerv3 ## our implementation
pip intstall -r requirements.txt
```

### 2. Config your path
* in `dreamerv3/run_env.py` and `dreamerv3-torch/run_env.py` :

```python
line 58 >> ROOT_PATH = "/home/ujs/TPAMI" # replace to your path
```


 * if your SUMO_HOME is **not** setup properly, try to set manually in `env/merge/autoGenTraffic.sh`

 ```shell
 if [ -z "${SUMO_HOME}" ]; then
        export SUMO_HOME="/openbayes/home/dreamer/lib/python3.12/site-packages/sumo"
    echo "SUMO_HOME was not set. Using default: ${SUMO_HOME}"
else
 ```
 ### 3. log your wandb and change in every `dreamer.py`

 ```python
 >>> line 274 
 def main(config):
    wandb.init(project="dreamer3", name="dreamer")
    tools.set_seed_everywhere(config.seed)
```
## Run the demo

### RiskDreamer:
```shell 
cd dreamerv3
python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

### Dreamerv3:
```shell
```shell 
cd dreamerv3-torch
python dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```

## Improve our algorithm

```python
>> line 146:def plan_action(self, post, em, is_training):
                planning_horizon = 10  

```