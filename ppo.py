"""Le cerveau de l'IA : un reseau de neurones entraine avec PPO (Proximal Policy Optimization)."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Yeux (CNN) + petit vecteur RAM (flags, badges, equipe, PV) -> tete acteur (quelle touche ?)
    + tete critique (combien de points a venir ?)."""

    def __init__(self, n_actions, in_channels, vec_size):
        super().__init__()
        self.eyes = nn.Sequential(                       # entree : 4 x 72 x 96 (3 images + carte visitee)
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),  # -> 32 x 17 x 23
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),          # -> 64 x 7 x 10
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),          # -> 64 x 5 x 8
            nn.Flatten(),
        )
        self.ram = nn.Sequential(nn.Linear(vec_size, 32), nn.ReLU())  # ce que l'ecran ne montre pas
        self.fc = nn.Sequential(nn.Linear(64 * 5 * 8 + 32, 512), nn.ReLU())
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, obs, vec, mask=None):
        features = self.fc(torch.cat([self.eyes(obs.float() / 255.0), self.ram(vec.float())], dim=1))
        logits = self.actor(features)
        if mask is not None:  # touches interdites (X / START sans Pokemon) : probabilite 0
            logits = logits.masked_fill(~mask, float("-inf"))
        return Categorical(logits=logits), self.critic(features).squeeze(-1)


class PPO:
    def __init__(self, n_actions, in_channels, vec_size, device, lr=2.5e-4, gamma=0.995, lam=0.95, clip=0.2,
                 epochs=4, minibatch=64, ent_coef=0.02, vf_coef=0.5):
        # ent_coef : bonus de hasard. A 0.01 l'entropie tombait vers 0.8 et l'IA se coinçait dans les coins.
        # gamma : a 0.98 l'IA ne "voyait" qu'environ 50 actions devant elle ; a 0.995, environ 200.
        self.net = ActorCritic(n_actions, in_channels, vec_size).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.device = device
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef

    def _tensors(self, obs, vec, mask):
        d = self.device
        return (torch.as_tensor(np.asarray(obs), device=d), torch.as_tensor(np.asarray(vec), device=d),
                torch.as_tensor(np.asarray(mask), device=d, dtype=torch.bool))

    @torch.no_grad()
    def act(self, obs, vec, mask):
        """Pour un lot d'observations (une par fenetre) : tire au sort une action pour chacune.
        Renvoie aussi les probabilites de chaque touche (pour l'affichage)."""
        dist, value = self.net(*self._tensors(obs, vec, mask))
        action = dist.sample()
        return (action.cpu().numpy(), dist.log_prob(action).cpu().numpy(), value.cpu().numpy(),
                dist.probs.cpu().numpy())

    @torch.no_grad()
    def value(self, obs, vec, mask):
        return self.net(*self._tensors(obs, vec, mask))[1].cpu().numpy()

    def advantages(self, rewards, values, dones, last_value):
        """GAE : pour chaque action, 'combien de points de plus que prevu' elle a rapporte.
        Tableaux de forme (temps, fenetres) : chaque fenetre est une partie independante."""
        rewards, values, dones = (np.asarray(x, dtype=np.float32) for x in (rewards, values, dones))
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            gae = delta + self.gamma * self.lam * not_done * gae
            adv[t] = gae
        return adv, adv + values

    def update(self, obs, vec, mask, actions, old_logp, advantages, returns):
        d = self.device
        obs, vec, mask = self._tensors(obs, vec, mask)
        actions = torch.as_tensor(actions, device=d)
        old_logp = torch.as_tensor(old_logp, device=d, dtype=torch.float32)
        adv = torch.as_tensor(advantages, device=d)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = torch.as_tensor(returns, device=d)

        stats = []
        for _ in range(self.epochs):
            for idx in torch.randperm(len(actions), device=d).split(self.minibatch):
                dist, value = self.net(obs[idx], vec[idx], mask[idx])  # meme masque qu'au moment de jouer
                ratio = (dist.log_prob(actions[idx]) - old_logp[idx]).exp()
                # Le "clip" de PPO : on ne change pas trop la politique d'un coup
                policy_loss = -torch.min(ratio * adv[idx],
                                         ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                value_loss = (value - returns[idx]).pow(2).mean()
                entropy = dist.entropy().mean()  # bonus de curiosite : garder un peu de hasard
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                stats.append((policy_loss.item(), value_loss.item(), entropy.item()))
        return np.mean(stats, axis=0)

    def save(self, path, extra=None):
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(), **(extra or {})}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["net"]
        own = self.net.state_dict()
        same = True

        if "eyes.7.weight" in state:
            # Ancien cerveau (sans vecteur RAM) : sa couche de 512 lisait seulement le CNN.
            # On la recopie, et les nouvelles entrees (vecteur RAM) partent a 0 -> meme comportement au depart.
            old_w = state.pop("eyes.7.weight")
            weight = torch.zeros_like(own["fc.0.weight"])
            weight[:, :old_w.shape[1]] = old_w
            state["fc.0.weight"], state["fc.0.bias"] = weight, state.pop("eyes.7.bias")
            state["ram.0.weight"], state["ram.0.bias"] = own["ram.0.weight"], own["ram.0.bias"]
            print("Cerveau v3 -> v4 : branche RAM ajoutee (le reste de l'apprentissage est conserve)")
            same = False

        old_n, new_n = state["actor.weight"].shape[0], own["actor.weight"].shape[0]
        if old_n != new_n:
            # Nouvelles touches : on garde tout ce qui a ete appris et on greffe des sorties neuves.
            # Leur biais = moyenne des anciennes -> ni favorisees ni ignorees au depart.
            assert new_n > old_n, "on ne peut qu'ajouter des touches a la fin de ACTIONS"
            weight, bias = own["actor.weight"].clone(), own["actor.bias"].clone()
            weight[:old_n] = state["actor.weight"]
            weight[old_n:] *= 0.01
            bias[:old_n] = state["actor.bias"]
            bias[old_n:] = state["actor.bias"].mean()
            state["actor.weight"], state["actor.bias"] = weight, bias
            print(f"Cerveau agrandi : {old_n} -> {new_n} touches (le reste de l'apprentissage est conserve)")
            same = False

        self.net.load_state_dict(state)
        if same:
            self.opt.load_state_dict(ckpt["opt"])
        # sinon l'optimiseur repart de zero : ses statistiques ne correspondent plus aux nouvelles couches
        return ckpt
