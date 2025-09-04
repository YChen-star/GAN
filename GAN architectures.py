
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ===== 0) Setup =====
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["figure.dpi"] = 130

# ===== 1) Data =====
def load_returns(path="CAC40 data.xlsx", col="Returns"):
    df = pd.read_excel(path, skiprows=2)
    if col not in df.columns:
        df.columns = ["Date","Close","Returns","Sorted Returns"]
    rets = pd.to_numeric(df[col], errors="coerce").dropna().values.astype("float64")
    return rets

rets = load_returns()
mu, sd = rets.mean(), rets.std(ddof=0) + 1e-12
z  = (rets - mu) / sd
to_real = lambda x: x * sd + mu
X = torch.from_numpy(z.astype("float32")).to(device)

# ===== 2) VaR/ES =====
def var_es_np(x: np.ndarray, a: float = 0.05, tail: str = "left"):
    q = np.quantile(x, a)
    if tail == "left":
        es = x[x <= q].mean() if (x <= q).any() else q
    elif tail == "right":
        es = x[x >= q].mean() if (x >= q).any() else q
    else:
        raise ValueError("tail must be 'left' or 'right'")
    return float(q), float(es)

# ===== 3) FZ Score S_alpha（Eq. 7, 修正版）=====
class TailScore(nn.Module):
    def __init__(self, alpha=0.05, Walpha=1.9):
        super().__init__()
        self.a = float(alpha)
        self.Wa = float(Walpha)

    def forward(self, v, e, x):
        a, Wa = self.a, self.Wa
        ind_le_v = (x <= v).float()
        term1 = (Wa / (2.0 * a)) * (ind_le_v - a) * (x**2 - v**2)
        term2 = (x < v).float() * e * (v - x)
        term3 = a * e * (e / 2.0 - v)
        return term1 + term2 + term3

def sort_vec(x1b: torch.Tensor) -> torch.Tensor:
    x1b = x1b + 1e-6 * torch.randn_like(x1b)
    return torch.sort(x1b.view(1, -1), dim=1).values

# ===== 4) Models =====
class Discriminator(nn.Module):
    def __init__(self, n_in, Walpha=1.9):
        super().__init__(); self.Wa=float(Walpha)
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),  nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2)
        )
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, svec):
        a, b = self.net(svec).squeeze(0)
        v = -F.softplus(a) 
        e = v * ( self.Wa + torch.sigmoid(b)*(1.0 - self.Wa) )
        return v, e

class Generator(nn.Module):
    def __init__(self, nz=128, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(nz, hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, inplace=True)
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.head_s  = nn.Linear(hidden, 1)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, z):
        h  = self.backbone(z)
        mu = self.head_mu(h).squeeze(1)
        s  = F.softplus(self.head_s(h)).squeeze(1) + 1e-3 
        eps = torch.randn_like(mu)
        return mu + s * eps

# ===== 5) 右尾目标 -> 训练方向(-X)左尾目标 =====
def right_to_left_targets(z_real: np.ndarray, alpha_right=0.975):
    q_r = np.quantile(z_real, alpha_right)
    es_r = z_real[z_real >= q_r].mean() if (z_real >= q_r).any() else q_r
    return float(-q_r), float(-es_r) 

# ===== 6) 训练（重新平衡损失函数）=====
def train_tail_gan_final(
    X_for_train: torch.Tensor,
    *, alpha_schedule=(0.05,), Walpha=1.9,
    v_t=None, e_t=None,
    nz=128, NB=192, epochs=2000,
    lr_g=1e-4, lr_d=1e-4,             # <- 修改点：新增 G 和 D 的学习率
    lam_base=14.0, n_D=4, print_every=200, eval_n=10000,
    tail_q=0.15, rho_tail=0.6,
    weight_g_aux=2.0, weight_g_main=5.0
):
    assert v_t is not None and e_t is not None
    N = X_for_train.numel()
    q = torch.quantile(X_for_train, tail_q)
    tail_idx = (X_for_train <= q).nonzero(as_tuple=True)[0]
    all_idx  = torch.arange(N, device=device)

    def sample_real():
        nt = int(NB * rho_tail); nr = NB - nt
        it = tail_idx[torch.randint(0, tail_idx.numel(), (nt,), device=device)]
        ir = all_idx[ torch.randint(0, N, (nr,), device=device) ]
        return X_for_train[torch.cat([it, ir], 0)]

    D = Discriminator(n_in=NB, Walpha=Walpha).to(device)
    G = Generator(nz=nz, hidden=128).to(device)
    S = TailScore(alpha=alpha_schedule[0], Walpha=Walpha).to(device)

    # 使用两个不同的学习率
    optD = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.9))
    optG = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
    
    real_mean = X_for_train.mean().detach()
    real_std  = X_for_train.std(unbiased=False).detach()

    for ep in range(1, epochs+1):
        lam = 40.0 if ep < epochs//4 else lam_base
        alphas = [0.05, 0.025] if ep < epochs//3 else [0.025, 0.01]

        x_real = sample_real()

        # ----- D -----
        for _ in range(n_D):
            z = torch.randn(NB, nz, device=device)
            x_fake = G(z).detach()

            LD = 0.0
            for a in alphas:
                S.a = a
                vf, ef = D(sort_vec(x_fake))
                vr, er = D(sort_vec(x_real))
                L_fake = S(vf, ef, x_fake).mean()
                L_real = S(vr, er, x_real).mean()
                LD += L_fake - lam * L_real
            optD.zero_grad(); (-LD).backward(); optD.step()

        # ----- G -----
        z = torch.randn(NB, nz, device=device)
        x_fake = G(z)

        LG_main = 0.0
        for a in alphas:
            S.a = a
            vf, ef = D(sort_vec(x_fake))
            LG_main += S(vf, ef, x_fake).mean()

        LG_aux = 5.0 * (x_fake.std(unbiased=False) - real_std).pow(2) + \
                 2.0 * (x_fake.mean() - real_mean).pow(2)
        
        LG = weight_g_main * LG_main + weight_g_aux * LG_aux

        optG.zero_grad(); LG.backward(); optG.step()

        if (ep % print_every == 0) or (ep == epochs):
            with torch.no_grad():
                z_eval = torch.randn(eval_n, nz, device=device)
                gen_train = G(z_eval).cpu().numpy()
                gen_eval  = to_real(-gen_train)
            vg_r, eg_r = var_es_np(gen_eval, 0.975, tail="right")
            print(f"[Tail-GAN] Ep {ep:4d} | RIGHT 97.5% VaR={vg_r:+.6f}, ES={eg_r:+.6f}")

    return G, D


# ===== 7) Run =====
def evaluate_generated(real_returns, gen_returns, title_prefix="Tail-GAN"):
    real = np.asarray(real_returns, dtype=float)
    gen  = np.asarray(gen_returns,  dtype=float)
    
    q_real = np.quantile(real, [0.025, 0.5, 0.975])
    q_gen  = np.quantile(gen,  [0.025, 0.5, 0.975])
    D, p   = ks_2samp(real, gen)

    print("\n=== Distribution summary ===")
    print(f"Quantiles (2.5%, 50%, 97.5%) -> Real: {q_real},   Gen: {q_gen}")
    print(f"KS test: D={D:.4f}, p-value={p:.4f}")

    # 直方图
    plt.figure(figsize=(12,4.5))
    bins = 60
    plt.hist(real, bins=bins, density=True, alpha=0.45, label="Real")
    plt.hist(gen,  bins=bins, density=True, alpha=0.45, label="Generated")
    for v in q_real: plt.axvline(v, color="C0", ls="--", lw=1)
    for v in q_gen:  plt.axvline(v, color="C1", ls=":",  lw=1)
    plt.title(f"{title_prefix}: Histogram with key quantiles")
    plt.xlabel("Return"); plt.ylabel("Density"); plt.legend()
    plt.grid(True, ls="--", alpha=0.3); plt.tight_layout(); plt.show()


def plot_lefttail_qq(real, gen, title="Tail-GAN", qmax=0.20):
    qs = np.linspace(0.001, qmax, 200)
    rq = np.quantile(real, qs)
    gq = np.quantile(gen,  qs)
    lim = [min(rq.min(), gq.min()) * 1.05, 0.0]
    plt.figure(figsize=(12, 4.6))
    plt.scatter(rq, gq, s=14, alpha=0.7)
    plt.plot(lim, lim, 'k--', lw=1)
    plt.xlabel("Real quantiles")
    plt.ylabel("Generated quantiles")
    plt.title(f"{title}: Left-tail QQ plot (Generated vs Real)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout(); plt.show()
    
if __name__ == "__main__":
    alpha_test = 0.975; tail_side = "right"
    X_train = -X if tail_side == "right" else X
    
    v_t_z, e_t_z = right_to_left_targets(z, alpha_right=0.975)
    v_t = torch.tensor(v_t_z, dtype=torch.float32, device=device)
    e_t = torch.tensor(e_t_z, dtype=torch.float32, device=device)
    
    vr_real, er_real = var_es_np(rets, alpha_test, tail=tail_side)
    
    G, D = train_tail_gan_final(
        X_train, alpha_schedule=(0.05,), Walpha=1.9,
        v_t=v_t, e_t=e_t,
        nz=128, NB=192, epochs=3000,
        lr_g=1e-5,           # ✅ 生成器学习率
        lr_d=1.5e-5,           # ✅ 判别器学习率
        lam_base=44.0, n_D=4, print_every=200, eval_n=10000,
        tail_q=0.50, rho_tail=0.6,
        weight_g_aux=2.0, weight_g_main=1.0
    )

    @torch.no_grad()
    def sample(G, n=500, nz=128):
        z = torch.randn(n, nz, device=device)
        return G(z).cpu().numpy()

    gen500_z = sample(G, 500); gen10k_z = sample(G, 10000)
    gen500 = to_real(-gen500_z); gen10k = to_real(-gen10k_z)
    
    shift = rets.mean() - gen10k.mean()
    gen500 += shift; gen10k += shift

    vg, eg = var_es_np(gen10k, alpha_test, tail=tail_side)
    print(f"\nVaR/ES@{alpha_test*100:.1f}%  Real: {vr_real:+.6f}/{er_real:+.6f} | Tail-GAN: {vg:+.6f}/{eg:+.6f}")
    
    evaluate_generated(rets, gen10k, title_prefix="Tail-GAN (final attempt)")
    plot_lefttail_qq(rets, gen10k, title="Tail-GAN")
    
    # === 新增代码：自动保存生成的500个数据 ===
    # 确保 gen500 是 NumPy 数组
    pd.DataFrame(gen500_z, columns=["Generated_Return"]).to_csv("generated_returns.csv", index=False)
    print(f"\nSuccessfully saved 500 generated returns")