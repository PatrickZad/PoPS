import torch
import torch.nn as nn
from pops.utils.events import get_event_storage
from pops.utils.visualizer import mat_heatmap


def ortho_penalty(t):
    return ((t @ t.T - torch.eye(t.shape[0]).cuda()) ** 2).mean() * 1e-6 #NOTE

def dis_sim_penalty(t1,t2):
    sim=torch.mm(t1,t2.T)
    return sim.mean()

def dis_orth_penalty(t1,t2):
    return ((t1 @ t2.T ) ** 2).mean()


class DAPromptPool(nn.Module):
    def __init__(
        self,
        orth_mu,
        emb_d,
        n_tasks,
        pool_size,
        num_prompts,
        num_layers,
        loss_weight,
        topk,
        vis_period,
        key_dim=768,):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self.top_k = topk
        # n_tasks==e_pool_size in dual-p
        self.loss_weight = loss_weight
        self._init_smart(pool_size, num_prompts, list(range(num_layers)))
        self.vis_period = vis_period

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f"e_p_{e}", p)
            setattr(self, f"e_k_{e}", k)
        self.orth_mu=orth_mu
        for e in self.e_layers:
            a = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f"e_a_{e}", a)
    def _init_smart(
        self, e_pool_size, num_e_prompts, e_layers
    ):
        self.task_id_bootstrap = False

        # prompt locations
        self.e_layers = e_layers

        # prompt pool size
        self.e_p_length = num_e_prompts
        self.e_pool_size = e_pool_size
        self.task_id_bootstrap = True
    def process_task_count(self, task_id):
        self.task_count = task_id

    def forward(self, x_query, vis_mark, train=False, task_id=None):
        B, nL, C = x_query.shape
        p_return = []
        p_loss = 0.0
        orth_loss=0.0
        vis_attn = []
        for l in range(nL):
            # e prompts
            lp = []
            if l in self.e_layers:
                K = getattr(self, f"e_k_{l}")  # 0 based indexing here
                p = getattr(self, f"e_p_{l}")  # 0 based indexing here
                A = getattr(self, f"e_a_{l}")  # 0 based indexing here
                n_A=nn.functional.normalize(A, dim=1)

                # cosine similarity to match keys/querries
                a_query = torch.einsum("bd,kd->bkd", x_query[:, l, :], A)
                # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_query, dim=2)
                cos_sim = torch.einsum("bkd,kd->bk", q, n_K)

                if train:
                    vis_attn.append(cos_sim.detach())
                    # dual prompt during training uses task id
                    loss = (
                            1.0
                            - cos_sim[
                                :,
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,
                            ]
                        ).sum()
                    o_loss=ortho_penalty(n_K[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])+ortho_penalty(n_A[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k,:])
                    P_ = (
                            p[
                                self.task_count
                                * self.top_k : (self.task_count + 1)
                                * self.top_k
                            ]
                            .flatten(0, 1)
                            .unsqueeze(0)
                            .expand(B, -1, -1)
                        )  # B, num_e_prompts*topk, d
                    p_loss += loss
                    orth_loss+=o_loss
                else:
                    per_task_sim=cos_sim.reshape(B,-1,self.top_k).sum(-1).sum(0)
                    task_id=torch.argmax(per_task_sim).item()
                    k_idx=torch.arange(task_id*self.top_k,(task_id+1)*self.top_k).unsqueeze(0).repeat(B,1)
                    P_ = p[k_idx]
                lp.append(P_)

            p_return.append(torch.cat(lp, dim=1))

        if self.vis_period > 0 and train:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                vis_img = mat_heatmap(torch.cat(vis_attn, dim=-1), vmin=-1.0, vmax=1.0)
                vis_img = (
                    torch.tensor(vis_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                )
                storage.put_image(f"batch/prompt_{vis_mark}", vis_img)
        p_loss =p_loss* self.loss_weight
        orth_loss=orth_loss*self.orth_mu
        if self.training:
            return torch.stack(p_return, dim=0), [p_loss,orth_loss]
        else:
            return torch.stack(p_return, dim=0).flatten(2, 3), [p_loss,orth_loss]

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(*dims, ortho=False):
    p = torch.nn.Parameter(torch.FloatTensor(*dims), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p


def tensor_prompt_value(*dims, ortho=False):
    return tensor_prompt(*dims, ortho=ortho).data

def build_stage_prompt_pool(prompt_cfg,stage_num_prompts,emb_d,num_layers,key_dim,vis_period):
        if stage_num_prompts == 0:
            prompt_stage=nn.Identity() # trivial impl
        else:
            prompt_stage = DAPromptPool(
                    orth_mu=prompt_cfg.ORTH_MU,
                    emb_d=emb_d,
                    n_tasks=prompt_cfg.NUM_TASKS,
                    pool_size=prompt_cfg.POOL_SIZE,
                    num_prompts=stage_num_prompts,
                    num_layers=num_layers,
                    topk=prompt_cfg.TOP_K,
                    loss_weight=prompt_cfg.LOSS_WEIGHT,
                    key_dim=key_dim,
                    vis_period=vis_period,
                )
           
            prompt_stage.process_task_count(prompt_cfg.CURRENT_TASK)
        return prompt_stage