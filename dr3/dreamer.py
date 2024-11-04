from torch import distributions
from torch import nn
import torch
from parallel import Parallel, Damy
import envs.wrappers as wrappers
import util
import models
import exploration as expl
import ruamel.yaml as yaml
import numpy as np
import argparse
import functools
import os
import pathlib
import sys
import wandb
import shutil

sys.path.append(str(pathlib.Path(__file__).parent))


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super().__init__()
        self._config = config
        self._logger = logger
        self._should_log = util.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = util.Every(batch_steps / config.train_ratio)
        self._should_pretrain = util.Once()
        self._should_reset = util.Every(config.reset_every)
        self._should_expl = util.Until(
            int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._world_model = models.WorldModel(
            obs_space, act_space, self._step, config)
        self._behavior_model = models.ImagBehavior(config, self._world_model)

        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._world_model = torch.compile(self._world_model)
            self._behavior_model = torch.compile(self._behavior_model)

        self._expl_behavior = dict(
            greedy=lambda: self._behavior_model,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(
                config, self._world_model, lambda feature, state, action: self._world_model.heads["reward"](feature).mean()),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._world_model.preprocess(obs)
        embed = self._world_model.encoder(obs)
        latent, _ = self._world_model.dynamics.obs_step(
            latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feature = self._world_model.dynamics.get_feature(latent)
        if not training:
            actor = self._behavior_model.actor(feature)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feature)
            action = actor.sample()
        else:
            actor = self._behavior_model.actor(feature)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._world_model._train(data)
        metrics.update(mets)
        start = post

        def reward(feature, state, action): return self._world_model.heads["reward"](
            self._world_model.dynamics.get_feature(state)
        ).mode()
        metrics.update(self._behavior_model._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update(
                {"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def make_dataset(episodes, config):
    generator = util.sample_episodes(episodes, config.batch_length)
    dataset = util.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    import uuid
    from run_env import Highway_env
    env = Highway_env(label=str(uuid.uuid1()))
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def setup_config(config):
    logdir_path = pathlib.Path(config.logdir).expanduser()

    config.traindir = config.traindir or logdir_path / "train_eps"
    config.evaldir = config.evaldir or logdir_path / "eval_eps"

    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    if logdir_path.exists() and logdir_path.is_dir():
        shutil.rmtree(logdir_path)
        logdir_path.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    return logdir_path


def main(config):
    wandb.init(project="compare", name=f"dreamer")
    util.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        util.enable_deterministic_run()

    logdir_path = setup_config(config=config)
    logger = util.Logger(logdir_path, step=0)

    print("Create envs.")
    train_eps_cache = util.load_episodes(
        config.traindir, limit=config.dataset_size)
    eval_eps_cache = util.load_episodes(config.evaldir, limit=1)

    def make(mode, id): return make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    train_envs = [Parallel(env, "process") for env in train_envs]
    eval_envs = [Parallel(env, "process") for env in eval_envs]

    act_space = train_envs[0].action_space
    print("Action Space", act_space)

    config.num_actions = act_space.n if hasattr(
        act_space, "n") else act_space.shape[0]

    state = None
    prefill_num = max(0, config.prefill - 0)
    print(f"Prefill dataset ({prefill_num} steps).")
    if hasattr(act_space, "discrete"):
        random_actor_buf = util.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )
    else:
        random_actor_buf = distributions.independent.Independent(
            distributions.uniform.Uniform(
                torch.Tensor(act_space.low).repeat(config.envs, 1),
                torch.Tensor(act_space.high).repeat(config.envs, 1),
            ),
            1,
        )

    def random_agent(o, d, s):
        action = random_actor_buf.sample()
        logprob = random_actor_buf.log_prob(action)
        return {"action": action, "logprob": logprob}, None

    # prefill state
    # state: step, episode, done, length, obs, agent_state, reward
    state = util.simulate(
        agent=random_agent,
        envs=train_envs,
        cache=train_eps_cache,
        directory=config.traindir,
        logger=logger,
        limit=config.dataset_size,
        total_steps=prefill_num,
    )
    logger.step += prefill_num * config.action_repeat
    print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps_cache, config)
    dreamer_agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    dreamer_agent.requires_grad_(requires_grad=False)

    # make sure eval will be executed once after config.steps
    while dreamer_agent._step < config.steps + config.eval_every:
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy_agent = functools.partial(
                dreamer_agent, training=False)
            util.simulate(
                eval_policy_agent,
                eval_envs,
                eval_eps_cache,
                config.evaldir,
                logger,
                is_eval=True,
                max_episodes=config.eval_episode_num,
            )
        print("Start training.")
        state = util.simulate(
            dreamer_agent,
            train_envs,
            train_eps_cache,
            config.traindir,
            logger,
            limit=config.dataset_size,
            total_steps=config.eval_every,
            state=state,
        )
        # items_to_save = {
        #     "agent_state_dict": dreamer_agent.state_dict(),
        #     "optims_state_dict": util.recursively_collect_optim_state_dict(dreamer_agent),
        # }
        # torch.save(items_to_save, logdir_path / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        util.recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = util.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))
