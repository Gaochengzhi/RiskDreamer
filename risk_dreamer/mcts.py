import torch
import torch.nn as nn
import numpy as np


class GumbelMCTS:
    def __init__(
        self,
        policy_network,
        value_network,
        dynamics_model,
        reward_model,
        max_simulations=50,
        num_samples=10,
        c_puct=1.0,
        discount_factor=0.99,
        device="cuda",
    ):
        """
        max_simulations (int): Maximum number of MCTS simulations to run.
        num_samples (int): Number of actions to sample from the policy.
        c_puct (float): Exploration constant balancing exploration and exploitation.
        discount_factor (float): Discount factor for future rewards.
        device (str): Device to run computations ('cpu' or 'cuda').
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.max_simulations = max_simulations
        self.num_samples = num_samples
        self.c_puct = c_puct
        self.discount_factor = discount_factor
        self.device = device
        self.root = None

    class TreeNode:
        def __init__(self, state, parent=None, prior=0.0):
            """
            Parameters:
                state (dict): State at this node.
                parent (TreeNode): Parent node.
                prior (float): Prior probability of the node (from the policy network).
            """
            self.state = state
            self.parent = parent
            self.children = {}
            self.visit_count = 0
            self.value_sum = 0.0
            self.prior = prior

        def value(self):
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count

        def expanded(self):
            return len(self.children) > 0

    def select_action(self, state):
        self.root = self.TreeNode(state=state, parent=None, prior=0.0)

        feat = self.dynamics_model.get_feat(state)

        with torch.no_grad():
            policy_dist = self.policy_network(feat)
            sampled_actions = policy_dist.sample(
                (self.num_samples,)
            )  # [num_samples, action_dim]
            log_probs = policy_dist.log_prob(
                sampled_actions
            )  # [num_samples, action_dim]
            log_probs = log_probs.sum(dim=-1)  # [num_samples]

            gumbel_noise = (
                torch.from_numpy(np.random.gumbel(size=log_probs.shape))
                .float()
                .to(self.device)
            )

            gumbel_logits = log_probs + gumbel_noise

            sorted_indices = torch.argsort(gumbel_logits, descending=True)
            top_actions = sampled_actions[sorted_indices]
            top_log_probs = log_probs[sorted_indices]

            action_probs = torch.exp(top_log_probs)
            priors = action_probs / action_probs.sum()
        for i in range(len(top_actions)):
            action = top_actions[i]
            prior = priors[i]
            self.root.children[i] = {
                "action": action,
                "node": self.TreeNode(state=None, parent=self.root, prior=prior.item()),
            }

        for _ in range(self.max_simulations):
            node = self.root

            simulated_state = {k: v.clone() for k, v in node.state.items()}
            search_path = [node]
            rewards = []

            while node.expanded():
                action, node = self.select_child(node)
                simulated_state, reward = self.simulate_action(
                    simulated_state, action)
                search_path.append(node)
                rewards.append(reward)

            value = self.evaluate_state(simulated_state)
            node.state = simulated_state
            self.backpropagate(search_path, value, rewards)

        visit_counts = {
            i: child["node"].visit_count for i, child in self.root.children.items()
        }
        best_child_index = max(visit_counts, key=visit_counts.get)
        best_action = self.root.children[best_child_index]["action"]
        best_action_logprob = top_log_probs[best_child_index].item()

        return best_action, torch.Tensor([best_action_logprob])

    def select_child(self, node):
        """
        Selects a child node based on the UCT formula in EfficientZero.
        """
        total_visits = sum(
            child["node"].visit_count for child in node.children.values()
        )
        sqrt_total = np.sqrt(total_visits + 1e-8)
        best_score = float('-inf')
        best_action = None
        best_node = None

        for i, child in node.children.items():
            action = child["action"]
            child_node = child["node"]

            # 计算UCT值
            q_value = child_node.value() if child_node.visit_count > 0 else node.value()  # 使用mean Q value
            u_score = (
                self.c_puct * child_node.prior *
                sqrt_total / (1 + child_node.visit_count) *
                (1.25 + np.log((total_visits + 19652 + 1) / 19652))
            )
            uct_score = q_value + u_score

            if uct_score > best_score:
                best_score = uct_score
                best_action = action
                best_node = child_node

        return best_action, best_node

    def select_child_b(self, node):
        """
        Selects a child node based on the PUCT formula.
        """
        total_visits = sum(
            child["node"].visit_count for child in node.children.values()
        )
        sqrt_total = np.sqrt(total_visits + 1e-8)
        action_scores = {}

        for i, child in node.children.items():
            action = child["action"]
            child_node = child["node"]
            u_score = (
                self.c_puct
                * child_node.prior
                * sqrt_total
                / (1 + child_node.visit_count)
            )
            q_value = child_node.value()
            action_scores[i] = q_value + u_score

        best_child_index = max(action_scores, key=action_scores.get)
        best_child = node.children[best_child_index]
        return best_child["action"], best_child["node"]

    def simulate_action(self, state, action):
        """
        Simulates taking an action using the dynamics model.
        Parameters:
            state (dict): Current state dictionary.
            action (Tensor): Action to apply.
        Returns:
            tuple: (next_state dict, reward)
        """
        # Use dynamics model to get the next state and reward
        with torch.no_grad():
            # Simulate the next state
            # Ensure action has batch dimension
            prior = self.dynamics_model.img_step(state, action)
            # Get features from the next state
            feat = self.dynamics_model.get_feat(prior)
            # Predict reward from the reward model
            reward = self.reward_model(feat).mode().item()
        return prior, reward

    def evaluate_state(self, state):
        """
        Evaluates the value of a state using the value network.
        """
        with torch.no_grad():
            feat = self.dynamics_model.get_feat(state)
            value = self.value_network(feat).mode().item()
        return value

    def backpropagate(self, search_path, value, rewards):
        """
        Propagates the evaluation value back up the tree, using soft min-max normalization.
        """
        min_q_value = float('inf')
        max_q_value = float('-inf')

        # 找到树中所有访问节点的最小值和最大值
        for node in search_path:
            q_values = [child["node"].value()
                        for child in node.children.values()]
            if q_values:  # 如果节点有孩子
                min_q_value = min(min_q_value, *q_values)
                max_q_value = max(max_q_value, *q_values)

        epsilon = 0.01  # 设置epsilon值避免过度自信
        for node, reward in zip(reversed(search_path), reversed(rewards + [0.0])):
            node.visit_count += 1
            node.value_sum += value

            # 更新值并应用Soft Min-Max
            if max_q_value > min_q_value:
                value = (value - min_q_value) / \
                    max(max_q_value - min_q_value, epsilon)
            value = reward + self.discount_factor * value

def plan_action_with_gumbel_mcts(self, post, embed, planning_horizon, risk_threshold, top_k, num_samples):
    # 初始化轨迹列表，每个轨迹包含：
    # 'post': 当前隐状态
    # 'actions': 动作序列列表
    # 'cumulative_reward': 累计奖励
    # 'cumulative_risk': 累计风险
    # 'logprobs': 动作对数概率列表
    # 'terminated': 是否终止（风险超过阈值）
    trajectories = [{
        'post': post,
        'actions': [],
        'cumulative_reward': 0.0,
        'cumulative_risk': 0.0,
        'logprobs': [],
        'terminated': False
    }]

    for t in range(planning_horizon):
        all_posts = []
        all_trajectories = []

        # 收集未终止的轨迹
        for traj in trajectories:
            if not traj['terminated']:
                all_posts.append(traj['post'])
                all_trajectories.append(traj)

        if not all_posts:
            break  # 所有轨迹都已终止

        # 将所有posts和embeds合并
        batched_post = torch.stack(all_posts, dim=0)  # [batch_size, post_dim]
        batched_embed = embed.expand(len(all_posts), *embed.shape[1:])  # [batch_size, embed_dim]

        # 获取动作分布
        feat = self._wm.dynamics.get_feat(batched_post)
        actor = self._task_behavior.actor(feat)
        dist = actor  # 假设为Categorical分布或具有logits属性

        # 获取logits和logprob
        logits = dist.logits  # [batch_size, action_dim]
        logprobs = F.log_softmax(logits, dim=-1)

        # 添加Gumbel噪声并选择Top-K动作
        gumbel_noise = -torch.empty_like(logits).exponential_().log()  # 生成Gumbel噪声
        noisy_logits = logits + gumbel_noise
        topk_values, topk_indices = torch.topk(noisy_logits, k=top_k, dim=-1)  # [batch_size, top_k]

        # 生成Top-K动作列表
        # actions: [batch_size * top_k, action_dim]
        actions = F.one_hot(topk_indices.view(-1), num_classes=logits.size(-1)).float()

        # 扩展batched_post和batched_embed
        expanded_post = batched_post.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, batched_post.size(-1))
        expanded_embed = batched_embed.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, batched_embed.size(-1))

        # 进行一步模拟
        next_post, _ = self._wm.dynamics.img_step(expanded_post, actions)
        feat = self._wm.dynamics.get_feat(next_post)

        # 计算风险和价值
        risk_dist = self._wm.heads["risk"](feat)
        risk = risk_dist.mode().squeeze(-1)  # [batch_size * top_k]
        value_dist = self._wm.heads["value"](feat)
        value = value_dist.mode().squeeze(-1)  # [batch_size * top_k]

        # 计算对应的logprob
        selected_logprobs = logprobs.gather(-1, topk_indices)  # [batch_size, top_k]
        selected_logprobs = selected_logprobs.view(-1)  # [batch_size * top_k]

        # 更新轨迹
        new_trajectories = []
        idx = 0
        for i, traj in enumerate(all_trajectories):
            if traj['terminated']:
                continue  # 跳过已终止的轨迹
            for k in range(top_k):
                current_idx = idx
                idx += 1
                instantaneous_risk = risk[current_idx].item()
                cumulative_risk = traj['cumulative_risk'] + instantaneous_risk
                cumulative_reward = traj['cumulative_reward'] + value[current_idx].item()
                terminated = instantaneous_risk > risk_threshold
                new_traj = {
                    'post': next_post[current_idx],
                    'actions': traj['actions'] + [actions[current_idx]],
                    'cumulative_reward': cumulative_reward,
                    'cumulative_risk': cumulative_risk,
                    'logprobs': traj['logprobs'] + [selected_logprobs[current_idx]],
                    'terminated': terminated
                }
                new_trajectories.append(new_traj)
        trajectories = new_trajectories

    # 筛选未终止的轨迹
    valid_trajectories = [traj for traj in trajectories if not traj['terminated']]

    if not valid_trajectories:
        # 如果没有可行的轨迹，采取默认策略
        feat = self._wm.dynamics.get_feat(post)
        actor = self._task_behavior.actor(feat)
        action = actor.mode()
        logprob = actor.log_prob(action)
        return action, logprob

    else:
        # 基于累计奖励和风险，选择最佳轨迹
        # 可以定义一个效用函数，例如：utility = cumulative_reward - beta * cumulative_risk
        beta = some_value  # 风险厌恶系数，需要根据具体情况设置
        def utility(traj):
            return traj['cumulative_reward'] - beta * traj['cumulative_risk']

        best_traj = max(valid_trajectories, key=utility)
        best_action = best_traj['actions'][0]  # 返回第一个动作
        best_logprob = best_traj['logprobs'][0]  # 返回对应的logprob

        return best_action, best_logprob