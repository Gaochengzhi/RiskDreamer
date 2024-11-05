import shutil
import wandb
from torch import distributions as torchd
from torch import nn
import torch
from parallel import Parallel, Damy
import envs.wrappers as wrappers
import tools
import models
import exploration as expl
import ruamel.yaml as yaml
import numpy as np
import argparse
import functools
import os
import pathlib
import sys


os.environ["WANDB_SILENT"] = "true"
sys.path.append(str(pathlib.Path(__file__).parent))


class RiskDreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super().__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(
            int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0

        self._dataset = dataset
        self.act_space = act_space
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)

        def reward(f, s, a):
            return self._wm.heads["reward"](f).mean()

        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        if config.compile and os.name != "nt":
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

    def __call__(self, obs, resets, state=None, training=True):
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
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    wandb.log({name: float(np.mean(values))})
                # wandb.log(self._metrics)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(resets)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, raw_obs, state, training):
        if state is None:
            post = action = None
        else:
            post, action = state
        obs = self._wm.preprocess(raw_obs)
        embed = self._wm.encoder(obs)  # shape 1, 64
        # latent : post = {"stoch": stoch, "deter": prior["deter"], **stats}
        post, _prior = self._wm.dynamics.obs_step(
            post, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            post["stoch"] = post["mean"]
        feat = self._wm.dynamics.get_feat(post)
        if not training:  # eval
            # action, logprob = self._task_behavior.mcts.select_action(latent)
            action, logprob = self.plan_action(post, embed, training)
            pass
            # actor = self._task_behavior.actor(feat)
            # action = actor.mode()
            # logprob = actor.log_prob(action)
        elif self._should_expl(self._step):
            action, logprob = self.plan_action(post, embed, training)
            # actor = self._expl_behavior.actor(feat)
            # action = actor.sample()
            # logprob = actor.log_prob(action)
        else:
            action, logprob = self.plan_action(post, embed, training)
            # actor = self._task_behavior.actor(feat)
            # action = actor.sample()
            # logprob = actor.log_prob(action)
        # action, logprob = self._task_behavior.mcts.select_action(latent)

        post = {k: v.detach() for k, v in post.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (post, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}

        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post

        def reward(f, s, a):
            return self._wm.heads["reward"](self._wm.dynamics.get_feat(s)).mode()

        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update(
                {"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def plan_actionb(self, post, em, is_training):
        planning_horizon = 10   # 规划的未来步数
        num_iterations = 5      # CEM的迭代次数
        num_samples = 100       # 每次迭代采样的样本数
        elite_ratio = 0.2       # 精英样本的比例
        risk_coefficient = 1.0  # 风险系数，用于权衡奖励和风险
        device = self._config.device

        action_size = self.act_space.shape[0]
        action_low = torch.tensor(self.act_space.low, device=device)
        action_high = torch.tensor(self.act_space.high, device=device)

        # 初始化动作分布参数（均值和标准差）
        action_mean = torch.zeros(planning_horizon, action_size, device=device)
        action_std = torch.ones(planning_horizon, action_size, device=device) * 0.5

        for iteration in range(num_iterations):
            # 从当前动作分布中采样动作序列
            action_sequences = torch.randn(num_samples, planning_horizon, action_size, device=device) * action_std + action_mean
            # 将动作裁剪到合法范围
            action_sequences = torch.clamp(action_sequences, action_low, action_high)

            # 评估每个动作序列的累计奖励和风险
            total_rewards = []
            total_risks = []
            for i in range(num_samples):
                actions = action_sequences[i]
                cumulative_reward, cumulative_risk = self.evaluate_action_sequence(post, actions)
                total_rewards.append(cumulative_reward)
                total_risks.append(cumulative_risk)

            total_rewards = torch.tensor(total_rewards, device=device)
            total_risks = torch.tensor(total_risks, device=device)

            # 动态调整风险系数（或阈值）
            # 这里我们根据累计风险的分布，动态计算风险系数
            # 例如：根据风险的标准差或其他统计量来调整风险系数
            risk_coefficient = self.dynamic_risk_adjustment(total_risks)

            # 计算综合评分（奖励和风险的权衡）
            scores = total_rewards - risk_coefficient * total_risks

            # 选择精英样本
            num_elites = max(1, int(elite_ratio * num_samples))
            elite_indices = torch.topk(scores, num_elites).indices
            elite_action_sequences = action_sequences[elite_indices]

            # 更新动作分布参数为精英样本的均值和标准差
            action_mean = elite_action_sequences.mean(dim=0)
            action_std = elite_action_sequences.std(dim=0) + 1e-6  # 防止标准差为0

        # 最终选择最优的动作序列
        best_action_sequence = action_mean
        best_action = best_action_sequence[0]  # 执行第一个动作

        # 由于我们直接使用CEM的均值作为最优动作序列，logprob可以设置为None
        best_logprob = None

        return best_action.unsqueeze(0), best_logprob
    def dynamic_risk_adjustment(self, total_risks):
    # 可以根据风险的统计量动态调整风险系数
    # 这里以风险的标准差为例
        risk_std = total_risks.std()
        risk_mean = total_risks.mode()

        # 定义风险系数为风险标准差和平均值的函数
        # 可以根据需要调整公式
        risk_coefficient = risk_mean / (risk_std + 1e-6)

        # 将风险系数限定在合理范围内
        risk_coefficient = torch.clamp(risk_coefficient, min=0.1, max=1.0)

        return risk_coefficient.item()

    def evaluate_action_sequence(self, post, actions):
        cumulative_rewards = []
        cumulative_risks = []
        state = {k: v.clone() for k, v in post.items()}

        for t, action in enumerate(actions):
            # 预测下一个状态
            next_state = self._wm.dynamics.img_step(state, action.unsqueeze(0))
            feat = self._wm.dynamics.get_feat(next_state)

            # 计算奖励和风险分布
            reward_dist = self._wm.heads["reward"](feat)
            risk_dist = self._wm.heads["risk"](feat)

            # 从分布中采样多次，以构建风险分布
            num_samples = 10
            reward_samples = reward_dist.sample((num_samples,))
            risk_samples = risk_dist.sample((num_samples,))

            # 累计奖励和风险
            cumulative_rewards.append(reward_samples.mean(dim=0).cpu())
            cumulative_risks.append(risk_samples.cpu())

            # 更新状态
            state = next_state

        # 将所有风险样本拼接
        cumulative_risks = torch.stack(cumulative_risks, dim=0)  # shape: [horizon, num_samples]
        total_risks = cumulative_risks.sum(dim=0)  # 对时间步求和

        # 计算CoVaR
        confidence_level = 0.95
        var = torch.quantile(total_risks, confidence_level)
        cvar = total_risks[total_risks >= var].mean()

        # 计算累计奖励
        total_reward = sum([r.item() for r in cumulative_rewards])

        return total_reward, cvar.item()
    # def evaluate_action_sequence(self, post, actions):
    #     cumulative_reward = 0.0
    #     cumulative_risk = 0.0
    #     state = {k: v.clone() for k, v in post.items()}

    #     for t, action in enumerate(actions):
    #         # 预测下一个状态
    #         next_state = self._wm.dynamics.img_step(state, action.unsqueeze(0))
    #         feat = self._wm.dynamics.get_feat(next_state)

    #         # 计算奖励和风险分布
    #         reward_dist = self._wm.heads["reward"](feat)
    #         risk_dist = self._wm.heads["risk"](feat)

    #         # 从分布中采样以估计奖励和风险
    #         reward = reward_dist.mode()  # 可以使用mean()或sample()
    #         risk = risk_dist.mode()

    #         # 累计奖励和风险（可根据需要增加折扣因子）
    #         cumulative_reward += reward.item()
    #         cumulative_risk += risk.item()

    #         # 更新状态
    #         state = next_state

    #     return cumulative_reward, cumulative_risk


    def plan_action(self, post, em, is_training):
            planning_horizon = 10  # 规划的未来步数
            risk_threshold = 0.8   # 风险阈值，可根据需要调整
            num_simulations = 50   # 轨迹数量
            risk_coefficient = 1.0  # 风险系数，用于综合评估
            device = self._config.device

            # 将后验状态扩展为并行的模拟数量
            post = {k: v.unsqueeze(0).expand(num_simulations, *v.shape)
                    for k, v in post.items()}

            # 初始化累计风险、价值和熵
            cumulative_risk = torch.zeros(num_simulations, 1).to(device)
            cumulative_value = torch.zeros(num_simulations, 1).to(device)
            cumulative_entropy = torch.zeros(num_simulations, 1).to(device)

            # 初始化轨迹存储
            trajectories = []

            for t in range(planning_horizon):
                feat = self._wm.dynamics.get_feat(post)

                if is_training and t == 0:
                    # 在训练时，初始步骤使用探索策略，增加探索
                    actor = self._expl_behavior.actor(feat)
                    actions = actor.sample()
                    logprobs = actor.log_prob(actions)
                else:
                    # 在评估或后续步骤，使用任务策略
                    actor = self._task_behavior.actor(feat)
                    if is_training:
                        actions = actor.sample()
                        logprobs = actor.log_prob(actions)
                    else:
                        actions = actor.mode()
                        logprobs = actor.log_prob(actions)

                # 预测下一个状态
                next_post = self._wm.dynamics.img_step(post, actions)

                # 计算风险和价值
                risk_dist = self._wm.heads["risk"](feat)
                value_dist = self._wm.heads["reward"](feat)
                risk = risk_dist.mode()
                value = value_dist.mode()
                risk_entropy = risk_dist.entropy()
                post_entropy = self._wm.dynamics.get_dist(post).entropy()

                # 更新累计风险、价值和熵
                cumulative_risk += risk.squeeze(-1)

                cumulative_value += value.squeeze(-1)
                cumulative_entropy += (risk_entropy + post_entropy)

                # 更新后验状态
                post = next_post

                # 记录轨迹
                trajectories.append({
                    'actions': actions,
                    'logprobs': logprobs,
                    'cumulative_risk': cumulative_risk.clone(),
                    'cumulative_value': cumulative_value.clone(),
                    'cumulative_entropy': cumulative_entropy.clone()
                })

            # 在循环结束后，根据风险阈值过滤轨迹
            final_cumulative_risk = trajectories[-1]['cumulative_risk']
            valid_mask = final_cumulative_risk < risk_threshold

            if valid_mask.sum() == 0:
                # 如果没有有效的轨迹，采取随机动作或采取最低风险的动作
                min_risk_idx = torch.argmin(final_cumulative_risk)
                best_action = trajectories[0]['actions'][min_risk_idx]
                best_logprob = trajectories[0]['logprobs'][min_risk_idx]
                return best_action, best_logprob

            # 从有效的轨迹中选择最佳动作
            final_cumulative_value = trajectories[-1]['cumulative_value'][valid_mask]
            final_cumulative_entropy = trajectories[-1]['cumulative_entropy'][valid_mask]
            scores = final_cumulative_value - risk_coefficient * \
                final_cumulative_risk[valid_mask] - 0.5*final_cumulative_entropy
            best_idx = torch.argmax(scores)

            # 获取最佳动作和对应的 logprob
            if len(trajectories)>1:
                valid_actions = trajectories[1]['actions'][valid_mask]
                valid_logprobs = trajectories[1]['logprobs'][valid_mask]
            else:
                valid_actions = trajectories[0]['actions'][valid_mask]
                valid_logprobs = trajectories[0]['logprobs'][valid_mask]
            best_action = valid_actions[best_idx]
            best_logprob = valid_logprobs[best_idx]

            return best_action.unsqueeze(0), best_logprob.unsqueeze(0)

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def make_env(config, mode, id):
    import uuid
    from run_env import Highway_env

    # env = Highway_env(label=str(uuid.uuid1()))
    env = Highway_env(label=str(uuid.uuid1()))
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env

import csv
csv_file_path = 'video_prediction_data.csv'
def main(config):
    wandb.init(project="dreamer3", name="dreamer")
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    if not config.save_model:
        shutil.rmtree(logdir, ignore_errors=True)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(logdir, 0)
    print("Create envs.")
    directory = config.traindir
    train_episode = tools.load_episodes(directory, limit=config.dataset_size)
    directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)

    def make(mode, id):
        return make_env(config, mode, id)

    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    train_envs = [Parallel(env, "process") for env in train_envs]
    eval_envs = [Parallel(env, "process") for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    prefill = max(0, config.prefill - count_steps(config.traindir))
    print(f"Prefill dataset ({prefill} steps).")
    if hasattr(acts, "discrete"):
        random_actor = tools.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )
    else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.Tensor(acts.low).repeat(config.envs, 1),
                torch.Tensor(acts.high).repeat(config.envs, 1),
            ),
            1,
        )

    def random_agent(o, d, s):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None

    state = tools.simulate(
        random_agent,
        train_envs,
        train_episode,
        config.traindir,
        logger,
        limit=config.dataset_size,
        steps=prefill,
    )
    logger.step += prefill * config.action_repeat
    print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_episode, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = RiskDreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists() and config.eval_mode:
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(
            agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False


    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 写入文件头
        writer.writerow(['states', 'model', 'error'])
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                eval_episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                # video_pred = agent._wm.video_pred(next(eval_dataset))
                truth, model, error = agent._wm.video_pred(next(eval_dataset))
                
                np.save('truth.npy', truth.cpu().numpy()) # 保存 model
                np.save('model.npy', model.cpu().numpy())
                # 保存 error
                np.save('error.npy', error.cpu().numpy())
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_episode,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception as e:
            print(e)


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
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
