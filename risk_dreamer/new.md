


```python
obs = self._wm.preprocess(raw_obs)
embed = self._wm.encoder(obs)


post, _ = self._wm.dynamics.obs_step(post, action, embed, obs["is_first"])
feat = self._wm.dynamics.get_feat(post)
post_entrop = self.dynamics.get_dist(post).entropy()
risk = self._wm.heads["risk"](feat).mode()
risk_entropy = self._wm.heads["risk"](feat).entropy()
risk_log_prob = self._wm.heads["risk"](feat).log_prob()
actor = self._task_behavior.actor(feat)
action = actor.sample(sample_shape=(self._config.num_samples,))
# eval
action = actor.mode()




metrics : model_loss, model_grad_norm, image_loss, reward_loss, risk_loss, cont_loss kl_free, dyn_loss reo_loss, prior_ent, post_ent 
```

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out),
                    min=self.absmax)).detach()
        return out
    
