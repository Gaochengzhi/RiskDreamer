import wandb
from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 10
n_episode = 10
evaluator_env_num = 1
continuous_action_space = True
K = 20  # num_of_sampled_actions
num_simulations = 20
update_per_collect = None
replay_ratio = 0.25
batch_size = 256
max_env_step = int(6e7)
reanalyze_ratio = 0.
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

bipedalwalker_cont_sampled_efficientzero_config = dict(
    exp_name=f'data_sez/bipedalwalker_cont_sampled_efficientzero_k{K}_ns{num_simulations}_upc{update_per_collect}-rr{replay_ratio}_rer{reanalyze_ratio}_seed0',
    env=dict(
        env_id='highway',
        env_type='normal',
        continuous=True,
        manually_discretization=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=29,
            action_space_size=2,
            continuous_action_space=continuous_action_space,
            num_of_sampled_actions=K,
            sigma_type='conditioned',
            model_type='mlp',
            # self_supervised_learning_loss=True,
            res_connection_in_dynamics=True,
            norm_type='LN',
        ),
        # (str) The path of the pretrained model. If None, the model will be initialized by the default model.
        model_path=None,
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        learning_rate=1e-4,
        cos_lr_scheduler=True,
        lr_piecewise_constant_decay=False,
        policy_entropy_loss_weight=5e-3,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_ratio=replay_ratio,
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
bipedalwalker_cont_sampled_efficientzero_config = EasyDict(
    bipedalwalker_cont_sampled_efficientzero_config)
main_config = bipedalwalker_cont_sampled_efficientzero_config

bipedalwalker_cont_sampled_efficientzero_create_config = dict(
    env=dict(
        type='highway',
        import_names=['run_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sampled_efficientzero',
        import_names=['lzero.policy.sampled_efficientzero'],
    ),
)
bipedalwalker_cont_sampled_efficientzero_create_config = EasyDict(
    bipedalwalker_cont_sampled_efficientzero_create_config
)
create_config = bipedalwalker_cont_sampled_efficientzero_create_config

if __name__ == "__main__":
    wandb.init(project='dreamer3', name='wuyang')
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0,
                 model_path=main_config.policy.model_path, max_env_step=max_env_step)
