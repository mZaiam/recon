import torch
import torch.nn as nn

class AUTOMAP(nn.Module):
    def __init__(self, n, ntheta=128, activation_fc=nn.Tanh(), activation_cv=nn.ReLU(), filters_cv=64):
        '''AUTOMAP network. 
        
        Args:
            n: int with size of images.
            ntheta: int for angle degrees used in Radon transform.
        '''
        
        super(AUTOMAP, self).__init__()
        
        self.n = n
        self.ntheta = ntheta
        
        self.fc = nn.Sequential(
            nn.Linear(n * ntheta, n**2), 
            activation_fc,
            nn.Linear(n**2, n**2),
            activation_fc,
        ) 
        
        self.cv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=filters_cv,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            activation_cv,
            nn.Conv2d(
                in_channels=filters_cv,
                out_channels=filters_cv,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            activation_cv,
            nn.ConvTranspose2d(
                in_channels=filters_cv,
                out_channels=1,
                kernel_size=9,
                stride=1,
                padding=4,
            ),
        )
        
    def forward(self, x):
        x_flattened = x.flatten(start_dim=1)
        fc_out = self.fc(x_flattened)
        fc_out_reshape = fc_out.reshape(-1, 1, self.n, self.n)
        cv_out = self.cv(fc_out_reshape)
        return cv_out.reshape(-1, self.n, self.n)
