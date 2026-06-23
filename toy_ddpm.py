from sklearn.datasets import make_moons
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch 



N = 2000
T = 200

# Creating dataset
x,y = make_moons(N,random_state=41,noise=0.1)

# Normalizing
x = (x - x.mean(axis=0)) / x.std(axis=0)

# Setting constants 
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
cum_alphas = torch.cumprod(alphas, dim=0)
sqrt_cumprod_alpha = torch.sqrt(cum_alphas)
sqrt_one_minus = torch.sqrt(1 - cum_alphas)


def q_sample(x_0,t,epsilon=None):
    if epsilon is None:
        epsilon = torch.randn_like(x_0) 

    sqrt_ab = sqrt_cumprod_alpha[t].view(-1, 1)   # (B,) -> (B,1)                                         
    sqrt_om = sqrt_one_minus[t].view(-1, 1)                                                
    return sqrt_ab * x_0 + sqrt_om * epsilon  



d = 64
def sinusoidal_embedding(t):
    """
    t: (B,)
    Returns:
    (B,d)
    """
    inv_freq = 10000 ** (
        -torch.arange(0, d, 2, device=t.device).float() / d
    )
    args = t[:, None].float() * inv_freq[None, :]

    emb = torch.cat(
        [torch.sin(args), torch.cos(args)],
        dim=1
    ) 
    return emb

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2+d,128)
        self.act1 = nn.SiLU()
        self.linear2 = nn.Linear(128,128)
        self.act2 = nn.SiLU()
        self.linear3 = nn.Linear(128,128)
        self.act3 = nn.SiLU()
        self.output = nn.Linear(128,2)


    def forward(self,x_t,t):
        temb = sinusoidal_embedding(t)
        inp = torch.cat([x_t,temb],dim=1) 
        inp = self.act1(self.linear1(inp))
        inp = self.act2(self.linear2(inp))
        inp = self.act3(self.linear3(inp))
        out = self.output(inp)
        return out 
    
x0 = torch.randn(4, 2)
t  = torch.randint(0, T, (4,))
print(q_sample(x0, t).shape)


# Training
x_tensor = torch.tensor(x, dtype=torch.float32)    

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

B = 256
num_steps = 10000
model.train()
losses = []
 

for step in range(num_steps):
    idx = torch.randint(0, N, (B,))      # random batch indexes
    x_0 = x_tensor[idx]                  
    t   = torch.randint(0, T, (B,))       
    eps = torch.randn_like(x_0)           # LABEL
    x_t = q_sample(x_0, t, eps)          
    pred = model(x_t, t)                  
    loss = loss_fn(pred, eps)            

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        losses.append(loss.item())
    if step % 200 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")

plt.plot(np.arange(len(losses)),losses,"b")
plt.xlabel("Step")
plt.ylabel("loss")
plt.grid(True,alpha=.4)
plt.show()



@torch.no_grad()                                
def sample(n):
    model.eval()
    x = torch.randn(n, 2)                      
    for ti in reversed(range(T)):                
        t = torch.full((n,), ti, dtype=torch.long)
        eps_theta = model(x, t)                    

        alpha_t     = alphas[ti]                  
        alpha_bar_t = cum_alphas[ti]
        beta_t      = betas[ti]

        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (x - coef * eps_theta) / torch.sqrt(alpha_t)

        if ti > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z     # add some noises 
        else:
            x = mean                              
    return x

samples = sample(2000).numpy()
plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5)
plt.xlim(-3, 3); plt.ylim(-3, 3)
plt.title("Generated samples")
plt.show()

def visualize_reverse_process(n):
    pics = []
    ts = [T,150,100,50,1]
    model.eval()
    x = torch.randn(n,2)
    for ti in reversed(range(T)):
        t = torch.full((n,), ti, dtype=torch.long)
        eps_theta = model(x, t)                    

        alpha_t     = alphas[ti]                  
        alpha_bar_t = cum_alphas[ti]
        beta_t      = betas[ti]

        coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = (x - coef * eps_theta) / torch.sqrt(alpha_t)

        if ti > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z     # add some noises 
        else:
            x = mean   
        if ti in ts:
            pics.append(x.detach().numpy())
    return pics

samples = visualize_reverse_process(2000)
fig, axses = plt.subplots(1,len(samples),figsize=(16,9))
axs = axses.flatten()
for i, pic in enumerate(samples):
    axs[i].scatter(pic[:, 0], pic[:, 1], s=3, alpha=0.5)
    axs[i].set_xlim(-3, 3); axs[i].set_ylim(-3, 3)
plt.title("Generated samples")
plt.show()