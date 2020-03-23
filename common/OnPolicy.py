import torch
import torch.nn as nn
import torch.nn.functional as F

class OnPolicy(nn.Module):
    def __init__(self):
        super(OnPolicy, self).__init__()
        
    def forward(self, x):
        raise NotImplementedError
        
    def act(self, x, deterministic=False):       
        logit, value = self.forward(x)                       
        probs = F.softmax(logit, dim=1)       
        
        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(num_samples=1)
        
        return action
    
    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)
        
        probs     = F.softmax(logit, dim=1)
        log_probs = F.log_softmax(logit, dim=1)
        
        action_log_probs = log_probs.gather(1, action)       

        entropy = -(probs * log_probs).sum(1).mean()
        
        return logit, action_log_probs, value, entropy

    def calculate_conv_output(self, img_dim, out_channels, kernel_size, stride=1, padding=0):        
        output_width = (img_dim[0] - kernel_size + 2*padding) // stride + 1
        output_height = (img_dim[1] - kernel_size + 2*padding) // stride + 1
        return [out_channels, output_width, output_height]
    
class OnPolicyPlay(OnPolicy):
    def __init__(self):
        super(OnPolicyPlay, self).__init__()

    def act(self, x, deterministic=False):
        logit, value, imagined_state, imagined_reward = self.forward(x)                       
        probs = F.softmax(logit, dim=1)       
        
        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(num_samples=1)
        
        return action, imagined_state, imagined_reward