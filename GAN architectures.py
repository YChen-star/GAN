import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

########Tail-GAN##########
# Setup
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["figure.dpi"] = 130

# 1) Data 
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

# 2) VaR/ES
def var_es_np(x: np.ndarray, a: float = 0.05, tail: str = "left"):
    q = np.quantile(x, a)
    if tail == "left":
        es = x[x <= q].mean() if (x <= q).any() else q
    elif tail == "right":
        es = x[x >= q].mean() if (x >= q).any() else q
    else:
        raise ValueError("tail must be 'left' or 'right'")
    return float(q), float(es)

# 3) Score S_alpha
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

# 4) Models
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

# 5)left tail
def right_to_left_targets(z_real: np.ndarray, alpha_right=0.975):
    q_r = np.quantile(z_real, alpha_right)
    es_r = z_real[z_real >= q_r].mean() if (z_real >= q_r).any() else q_r
    return float(-q_r), float(-es_r) 

# 6) training
def train_tail_gan_final(
    X_for_train: torch.Tensor,
    *, alpha_schedule=(0.05,), Walpha=1.9,
    v_t=None, e_t=None,
    nz=128, NB=192, epochs=2000, lr=1e-4,
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

    optD = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    optG = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    
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

# 7) Run 
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
    
    GT, D = train_tail_gan_final(
        X_train, alpha_schedule=(0.05,), Walpha=1.9,
        v_t=v_t, e_t=e_t,
        nz=128, NB=192, epochs=5000, lr=1e-5,
        lam_base=14.0, n_D=4, print_every=1000, eval_n=10000,
        tail_q=0.50, rho_tail=0.6,
        weight_g_aux=2.0, weight_g_main=1.0 
    )

    @torch.no_grad()
    def sample(G, n=500, nz=128):
        z = torch.randn(n, nz, device=device)
        return G(z).cpu().numpy()

    gen500_z = sample(GT, 500); gen10k_z = sample(GT, 10000)
    gen500 = to_real(-gen500_z); gen10k = to_real(-gen10k_z)
    
    pd.DataFrame(gen500, columns=["Generated_Return"]).to_csv("generated_returns_500.csv", index=False)
    
    shift = rets.mean() - gen10k.mean()
    gen10k_plot = gen10k + shift

    vg, eg = var_es_np(gen10k_plot, alpha_test, tail=tail_side)
    print(f"\nVaR/ES@{alpha_test*100:.1f}%  Real: {vr_real:+.6f}/{er_real:+.6f} | Tail-GAN: {vg:+.6f}/{eg:+.6f}")
    
    evaluate_generated(rets, gen10k_plot, title_prefix="Tail-GAN (final attempt)")
    plot_lefttail_qq(rets, gen10k_plot, title="Tail-GAN")
    
    
    

########WGAN-GP##########
# 1）Setup
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
plt.rcParams["figure.dpi"] = 130

file_path = "CAC40 data.xlsx"
df = pd.read_excel(file_path, skiprows=2)
df.columns = ["Date","Close","Returns","Sorted Returns"]
df["Returns"] = pd.to_numeric(df["Returns"], errors="coerce")
rets = df["Returns"].dropna().reset_index(drop=True)

NORM_MODE = "z"

if NORM_MODE == "z":
    mu, sd = rets.mean(), rets.std(ddof=0)
    x_std = (rets - mu) / sd
    to_real = lambda x: x * sd + mu
elif NORM_MODE == "robust":
    med = rets.median()
    mad = (np.abs(rets - med)).median() + 1e-8
    scale = 1.4826 * mad
    x_std = (rets - med) / scale
    to_real = lambda x: x * scale + med
else:
    x_std = rets.copy()
    to_real = lambda x: x

X = torch.from_numpy(x_std.values.astype("float32")).unsqueeze(1)

# 2) Model
class Generator(nn.Module):
    def __init__(self, nz=64, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,128), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, out_dim)
        )
        self.apply(self._init)
        last = self.net[-1]
        nn.init.xavier_uniform_(last.weight)
        nn.init.zeros_(last.bias)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

class Critic(nn.Module):
    def __init__(self, in_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,256),    nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
    
# 3) Gradient Penalty
def gradient_penalty(D, real, fake, target=1.0):
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, device=real.device).expand_as(real)
    xhat = eps * real + (1 - eps) * fake
    xhat.requires_grad_(True)
    d = D(xhat)
    grads = torch.autograd.grad(d, xhat, torch.ones_like(d),
                                create_graph=True, retain_graph=True)[0]
    gnorm = grads.view(bsz, -1).norm(2, dim=1)
    gp = ((gnorm - target) ** 2).mean()
    return gp, gnorm.detach()

# 4) Training
def train_wgangp(
    X_real,
    nz=64, batch_size=256, epochs=2500,
    n_critic=3,                  
    lr_G=3e-4, lr_D=1e-4, betas=(0.0, 0.9),

    lambda_gp=1.0, gp_target=1.0, gp_adapt_lr=0.02,
    lambda_gp_min=0.5, lambda_gp_max=50.0,

    inst_noise_start=0.03, inst_noise_end=0.005,   
    grad_clip=10.0,
    ema_decay=0.9997,          

    moment_penalty_w=0.05,    
    penalty_ramp=1200,           

    device="cpu"
):

    dl = DataLoader(TensorDataset(X_real), batch_size=batch_size,
                    shuffle=True, drop_last=True)

    G, D = Generator(nz, X_real.size(1)).to(device), Critic(X_real.size(1)).to(device)
    optG = optim.Adam(G.parameters(), lr=lr_G, betas=betas)
    optD = optim.Adam(D.parameters(), lr=lr_D, betas=betas)

    G_ema = Generator(nz, X_real.size(1)).to(device)
    G_ema.load_state_dict(G.state_dict())
    for p in G_ema.parameters():
        p.requires_grad_(False)

    logs = {"w": [], "loss_D": [], "loss_G": [], "gp": [], "gmean": [], "lam": []}

    norm_mode = globals().get("NORM_MODE", "z")

    for ep in range(1, epochs + 1):
        t = (ep - 1) / max(epochs - 1, 1)
        sigma = inst_noise_start + (inst_noise_end - inst_noise_start) * t

        if moment_penalty_w > 0.0 and penalty_ramp and penalty_ramp > 0:
            progress = min(ep / float(penalty_ramp), 1.0)
            mm_w = moment_penalty_w * progress
        else:
            mm_w = 0.0

        for (real,) in dl:
            real = real.to(device)
            bsz = real.size(0)

            for _ in range(n_critic):
                z = torch.randn(bsz, nz, device=device)
                with torch.no_grad():
                    fake = G(z)

                if sigma > 0:
                    real_in = real + torch.randn_like(real) * sigma
                    fake_in = fake + torch.randn_like(fake) * sigma
                else:
                    real_in, fake_in = real, fake

                d_real = D(real_in).mean()
                d_fake = D(fake_in).mean()
                w = d_real - d_fake

                gp, gnorm = gradient_penalty(D, real_in, fake_in, gp_target)
                lossD = (d_fake - d_real) + lambda_gp * gp  

                optD.zero_grad()
                lossD.backward()
                optD.step()

                logs["w"].append(w.item())
                logs["loss_D"].append(lossD.item())
                logs["gp"].append(gp.item())
                logs["gmean"].append(gnorm.mean().item())
                logs["lam"].append(lambda_gp)

                if gp_adapt_lr and gp_adapt_lr > 0.0:
                    lambda_gp *= np.exp(gp_adapt_lr * (gnorm.mean().item() - gp_target))
                    if (lambda_gp_min is not None) or (lambda_gp_max is not None):
                        lo = -float("inf") if lambda_gp_min is None else lambda_gp_min
                        hi =  float("inf") if lambda_gp_max is None else lambda_gp_max
                        lambda_gp = float(np.clip(lambda_gp, lo, hi))

            z = torch.randn(bsz, nz, device=device)
            fake = G(z)
            if sigma > 0:
                fake = fake + torch.randn_like(fake) * sigma

            loss_adv = -D(fake).mean()

            if norm_mode != "none" and mm_w > 0:
                mu_pen = (fake.mean()) ** 2
                std_pen = (fake.std(unbiased=False) - 1.0) ** 2
                lossG = loss_adv + mm_w * (mu_pen + std_pen)
            else:
                lossG = loss_adv

            optG.zero_grad()
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            optG.step()
            logs["loss_G"].append(lossG.item())

            ema_now = 0.99 if ep < 500 else ema_decay
            with torch.no_grad():
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.mul_(ema_now).add_(p, alpha=1 - ema_now)

        if ep % 200 == 0 or ep == epochs:
            with torch.no_grad():
                z = torch.randn(8192, nz, device=device)
                samp = G_ema(z).cpu().numpy().reshape(-1)
            ep_batches = len(dl)
            gbar = (np.mean(logs["gmean"][-ep_batches:]) if len(logs["gmean"]) >= ep_batches else np.nan)
            print(f"[WGANGP] Epoch {ep:4d} | mean={samp.mean():+.3f}, std={samp.std(ddof=0):.3f} "
                  f"| λ_gp={lambda_gp:.3f} | E||∇||={gbar:.2f}")

    return G, D, G_ema, logs


@torch.no_grad()
def sample_real_space(G_smpl, n, nz, to_real, device):
    z = torch.randn(n, nz, device=device)
    x = G_smpl(z).cpu().numpy().reshape(-1)
    return to_real(x)

def movavg(x, k=50):
    x = np.asarray(x, float)
    if len(x)<k: return x
    c = np.cumsum(np.insert(x,0,0.0))
    return (c[k:] - c[:-k]) / k

def var_es(x, alpha=0.05):
    x = np.asarray(x, float)
    q = np.quantile(x, alpha)
    es = x[x <= q].mean() if np.any(x <= q) else q
    return q, es

def plot_hist(real, gen, title="WGAN-GP"):
    plt.figure(figsize=(12,4.6))
    bins = 80
    plt.hist(real, bins=bins, density=True, alpha=.55, label="Real")
    plt.hist(gen,  bins=bins, density=True, alpha=.55, label="Generated")
    plt.legend(); plt.xlabel("Return"); plt.ylabel("Density")
    plt.title(rf"{title}: Histogram — Real vs Generated")
    plt.grid(True, ls="--", alpha=.35); plt.tight_layout(); plt.show()

def plot_lefttail_qq(real, gen, title="WGAN-GP", qmax=0.20):
    qs = np.linspace(0.001, qmax, 200)
    rq = np.quantile(real, qs); gq = np.quantile(gen, qs)
    lim = [min(rq.min(), gq.min())*1.05, 0.0]
    plt.figure(figsize=(12,4.6))
    plt.scatter(rq, gq, s=14, alpha=0.7)
    plt.plot(lim, lim, 'k--', lw=1)
    plt.xlabel("Real quantiles"); plt.ylabel("Generated quantiles")
    plt.title(rf"{title}: Left-tail QQ plot (Generated vs Real)")
    plt.grid(True, ls="--", alpha=.35); plt.tight_layout(); plt.show()

def plot_lefttail_cdf(real, gen, title="WGAN-GP", qmax=0.20):
    r = np.sort(real); g = np.sort(gen)
    r = r[r <= np.quantile(real, qmax)]
    g = g[g <= np.quantile(gen,  qmax)]
    Fr = np.arange(1, len(r)+1) / len(real)
    Fg = np.arange(1, len(g)+1) / len(gen)
    plt.figure(figsize=(12,4.6))
    plt.semilogy(r, Fr[:len(r)], label="Real CDF (left tail)")
    plt.semilogy(g, Fg[:len(g)], label="Gen CDF (left tail)")
    plt.xlabel("Return (left tail)"); plt.ylabel("P(X ≤ x)")
    plt.title(f"{title}: Left-tail CDF (log scale)")
    plt.grid(True, ls="--", which="both", alpha=.35); plt.legend()
    plt.tight_layout(); plt.show()

def print_summary(real, gen, title="WGAN-GP"):
    r, g = pd.Series(real), pd.Series(gen)
    print(f"\n=== {title} Summary ===")
    print(f"Real: mean={r.mean():+.6f}, std={r.std(ddof=0):.6f}, skew={r.skew():+.3f}, kurt(excess)={r.kurt():+.3f}")
    print(f"Gen : mean={g.mean():+.6f}, std={g.std(ddof=0):.6f}, skew={g.skew():+.3f}, kurt(excess)={g.kurt():+.3f}")
    for a in (0.05, 0.025, 0.01):
        vr, er = var_es(real, a); vg, eg = var_es(gen, a)
        print(f"alpha={a:>5.3f} | VaR real={vr:+.6f}, gen={vg:+.6f} | ES real={er:+.6f}, gen={eg:+.6f}")

def plot_training_logs(logs, title="WGANGP"):
    plt.figure(figsize=(12,4))
    plt.plot(movavg(logs["w"], 50))
    plt.title(f"{title}: Estimated Wasserstein Distance")
    plt.grid(True, ls="--", alpha=.35); plt.tight_layout(); plt.show()

    plt.figure(figsize=(12,4))
    plt.plot(movavg(logs["loss_D"], 50), label="Critic loss")
    plt.plot(movavg(logs["loss_G"], 50), label="Generator loss")
    plt.plot(movavg(logs["gp"],     50), label="Gradient penalty")
    plt.plot(movavg(logs["gmean"],  50), label="E||∇||")
    plt.legend(); plt.title(f"{title}: Training Curves")
    plt.grid(True, ls="--", alpha=.35); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    nz = 128
    G, D, G_ema, logs = train_wgangp(
    X.to(device),
    nz=128,
    batch_size=128,          
    epochs=3000,
    n_critic=7,       
    lr_G=1e-4, lr_D=2e-4, 
    betas=(0.0, 0.9),

    lambda_gp=1.0, gp_target=1.0, gp_adapt_lr=0.0,
    lambda_gp_min=1.0, lambda_gp_max=1.0,

    inst_noise_start=0.0, inst_noise_end=0.0,

    moment_penalty_w=0.02, penalty_ramp=600,
    grad_clip=5.0,
    ema_decay=0.999,
    device=device
)


    real = rets.values
    gen500 = sample_real_space(G_ema, 500,    nz, to_real, device)
    gen10k = sample_real_space(G_ema, 10_000, nz, to_real, device)

    plot_training_logs(logs, "WGANGP")
    plot_hist(real, gen500, "WGANGP")
    plot_lefttail_qq(real, gen10k, "WGANGP", qmax=0.20)
    plot_lefttail_cdf(real, gen10k, "WGANGP", qmax=0.20)
    print_summary(real, gen10k, "WGANGP")

    out_csv = "generated_returns_WGANGP_500.csv"
    pd.DataFrame({"Generated_Return": gen500}).to_csv(out_csv, index=False)
    print(f"\nSaved 500 generated returns -> {os.path.abspath(out_csv)}")







########WGAN-GP##########
file_path = "CAC40 data.xlsx"
df = pd.read_excel(file_path, skiprows=2)
df.columns = ['Date', 'Closing Price', 'Returns', 'Sorted Returns']

df['Returns'] = pd.to_numeric(df['Returns'], errors='coerce')
returns = df['Returns'].dropna()

mean = returns.mean()
std = returns.std()
returns_norm = (returns - mean) / std
returns_tensor = torch.from_numpy(returns_norm.values.astype('float32')).unsqueeze(1)


# 2）WGAN Model
class Generator1(nn.Module):
    def __init__(self, noise_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, z):
        return self.net(z)

class Critic1(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

# 3）WGAN training function
def train_wgan(X_real, noise_dim=10, batch_size=64, n_epochs=2000, n_critic=5, 
               lr=1e-4, clip_value=0.01, device="cpu"):
    data_loader = DataLoader(TensorDataset(X_real), batch_size=batch_size, shuffle=True)
    G = Generator1(noise_dim, X_real.size(1)).to(device)
    C = Critic1(X_real.size(1)).to(device)
    optim_G = optim.RMSprop(G.parameters(), lr=lr)
    optim_C = optim.RMSprop(C.parameters(), lr=lr)
    for epoch in range(n_epochs):
        for i, (real_data,) in enumerate(data_loader):
            real_data = real_data.to(device)
            for _ in range(n_critic):
                z = torch.randn(real_data.size(0), noise_dim, device=device)
                fake_data = G(z).detach()
                loss_C = -(C(real_data).mean() - C(fake_data).mean())
                optim_C.zero_grad()
                loss_C.backward()
                optim_C.step()
                for p in C.parameters():
                    p.data.clamp_(-clip_value, clip_value)
            z = torch.randn(real_data.size(0), noise_dim, device=device)
            fake_data = G(z)
            loss_G = -C(fake_data).mean()
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
        if epoch % 100 == 0 or epoch == n_epochs-1:
            print(f"Epoch {epoch}: Critic Loss: {loss_C.item():.4f}, Generator Loss: {loss_G.item():.4f}")
    return G

def generate_samples(G, n_samples, noise_dim=10, mean=0.0, std=1.0, device="cpu"):
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim, device=device)
        fake_samples = G(z)
    fake_returns = fake_samples.cpu().numpy().reshape(-1) * std + mean
    return fake_returns

def var_es1(x, alpha=0.01):
    x = np.asarray(x, float)
    var = np.quantile(x, alpha)
    es  = x[x <= var].mean() if (x <= var).any() else var
    return var, es

def describe(x):
    x = np.asarray(x, float)
    qs = np.quantile(x, [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])
    return dict(
        mean=float(x.mean()),
        std=float(x.std(ddof=1)),
        quantiles=qs
    )

def evaluate_generated(real_returns, gen_returns, title_prefix="WGAN"):
    real = np.asarray(real_returns, float)
    gen  = np.asarray(gen_returns,  float)

    rdesc, gdesc = describe(real), describe(gen)
    alphas = [0.05, 0.025, 0.01]

    print("\n=== Summary ===")
    print(f"Real: mean={rdesc['mean']:.6f}, std={rdesc['std']:.6f}")
    print(f"Gen : mean={gdesc['mean']:.6f}, std={gdesc['std']:.6f}")
    for a in alphas:
        r_var, r_es = var_es1(real, a)
        g_var, g_es = var_es1(gen,  a)
        print(f"alpha={a:>6.3f} | VaR  real={r_var:+.6f}, gen={g_var:+.6f}  | "
              f"ES real={r_es:+.6f}, gen={g_es:+.6f}")

    plt.figure(figsize=(12, 4.2))
    bins = 80
    plt.hist(real, bins=bins, density=True, alpha=0.45, label="Real")
    plt.hist(gen,  bins=bins, density=True, alpha=0.45, label="Generated")
    plt.title(f"{title_prefix}: Histogram — Real vs Generated")
    plt.xlabel("Return"); plt.ylabel("Density"); plt.legend()
    plt.grid(True, ls="--", alpha=0.3); plt.tight_layout(); plt.show()

    qs = np.linspace(0.001, 0.20, 200)
    rq = np.quantile(real, qs)
    gq = np.quantile(gen,  qs)
    lim = [min(rq.min(), gq.min()), max(rq.max(), gq.max())]
    plt.figure(figsize=(12, 4.2))
    plt.scatter(rq, gq, s=14, alpha=0.7)
    plt.plot(lim, lim, 'k--', lw=1)
    plt.title(f"{title_prefix}: Left-tail QQ plot (Generated vs Real)")
    plt.xlabel("Real quantiles"); plt.ylabel("Generated quantiles")
    plt.grid(True, ls="--", alpha=0.3); plt.tight_layout(); plt.show()

    xgrid = np.linspace(real.min(), np.quantile(real, 0.30), 200)
    def ecdf_le(x, grid): 
        x = np.asarray(x, float)
        return np.searchsorted(np.sort(x), grid, side='right') / x.size

    Fc_real = ecdf_le(real, xgrid)
    Fc_gen  = ecdf_le(gen,  xgrid)

    plt.figure(figsize=(12, 4.2))
    plt.plot(xgrid, Fc_real, label="Real CDF (left tail)")
    plt.plot(xgrid, Fc_gen,  label="Gen CDF (left tail)")
    plt.yscale("log") 
    plt.title(f"{title_prefix}: Left-tail CDF (log scale)")
    plt.xlabel("Return (left tail)"); plt.ylabel("P(X ≤ x)")
    plt.grid(True, ls="--", alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()
    
    
 # GAN architectures Comparison   
def plot_var_es_diff(real_returns, models_dict, alpha=0.01, n_models=100, n_samples=500, noise_dim=10, mean=0.0, std=1.0, device="cpu"):
    import matplotlib.pyplot as plt
    import numpy as np

    real_returns = np.asarray(real_returns, float)
    model_names = list(models_dict.keys())

    var_diffs = {name: [] for name in model_names}
    es_diffs  = {name: [] for name in model_names}

    for name in model_names:
        G = models_dict[name]
        dim = noise_dim[name] if isinstance(noise_dim, dict) else noise_dim
        G.eval()
        with torch.no_grad():
            for _ in range(n_models):
                z = torch.randn(n_samples, dim, device=device)
                if name == 'Tail-GAN':
                    fake = -G(z).cpu().numpy().reshape(-1) * std + mean
                else:
                    fake = G(z).cpu().numpy().reshape(-1) * std + mean

                var_real, es_real = var_es(real_returns, alpha)
                var_fake, es_fake = var_es(fake, alpha)

                var_diffs[name].append(abs(var_real - var_fake))
                es_diffs[name].append(abs(es_real - es_fake))

    # Plot VaR Difference
    plt.figure(figsize=(7, 5))
    for name in model_names:
        plt.plot(sorted(var_diffs[name]), label=name)
    plt.title(f"VaR Difference Between Real and Generated Data (α={alpha})")
    plt.xlabel("Model Index"); plt.ylabel("VaR Difference"); plt.legend()
    plt.grid(True, ls="--", alpha=0.5); plt.tight_layout(); plt.show()

    # Plot ES Difference
    plt.figure(figsize=(7, 5))
    for name in model_names:
        plt.plot(sorted(es_diffs[name]), label=name)
    plt.title(f"ES Difference Between Real and Generated Data (α={alpha})")
    plt.xlabel("Model Index"); plt.ylabel("ES Difference"); plt.legend()
    plt.grid(True, ls="--", alpha=0.5); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    G = train_wgan(returns_tensor, noise_dim=10, batch_size=64, n_epochs=2000, device="cpu")
    generated_returns = generate_samples(G, 500, noise_dim=10, mean=mean, std=std)
    generated_returns10k = generate_samples(G, 10000, noise_dim=10, mean=mean, std=std)
    pd.DataFrame(generated_returns, columns=["Generated_Return"]).to_csv("generated_returns.csv", index=False)
    evaluate_generated(returns.values, generated_returns10k, title_prefix="WGAN")

    

    models_dict = {
    'WGAN':    G,
    'WGAN-GP': G_ema,
    'Tail-GAN': GT,
    }
    noise_dims = {
        'WGAN': 10,
        'WGAN-GP': 128,
        'Tail-GAN': 128
    }
    
    plot_var_es_diff(returns.values, models_dict, alpha=0.025, n_models=100, n_samples=500,
                 noise_dim=noise_dims, mean=mean, std=std, device="cpu")


