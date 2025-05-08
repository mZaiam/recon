import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        cv_channels=[1, 16, 32, 64, 128, 256, 512],
        cv_kernel=(3, 3),
        cv_stride=2,
        cv_padding=(1, 1),
        cv_activation=nn.ReLU(),
        lin_num_neurons=[256, 128],
        lin_activation=nn.ReLU(),
        input_shape=(128, 128),
    ):
        super(Encoder, self).__init__()
        
        cv_layers = []
        for i in range(len(cv_channels) - 1):
            cv_layers.extend([
                nn.Conv2d(
                    in_channels=cv_channels[i], 
                    out_channels=cv_channels[i + 1],
                    kernel_size=cv_kernel,
                    stride=cv_stride,
                    padding=cv_padding,
                ),
                nn.BatchNorm2d(cv_channels[i + 1]),
                cv_activation,
            ])
                
        self.cv = nn.Sequential(*cv_layers)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, *input_shape)
            conv_output_shape = self.cv(dummy).squeeze().shape
            lin_input_neurons = self.cv(dummy).flatten().shape[0]
                        
        lin_layers = [
            nn.Linear(
                    in_features=lin_input_neurons, 
                    out_features=lin_num_neurons[0],
                ),
                lin_activation,
        ]
        
        for i in range(len(lin_num_neurons)):
            if i == len(lin_num_neurons) - 1:
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[-1], 
                        out_features=latent_dim,
                    ),
                ])
            else:
                lin_layers.extend([
                    nn.Linear(
                        in_features=lin_num_neurons[i], 
                        out_features=lin_num_neurons[i + 1],
                    ),
                    lin_activation,
                ])
        
        self.lin = nn.Sequential(*lin_layers)
            
    def forward(self, x):
        cv_out = self.cv(x)
        cv_out = cv_out.view(cv_out.size(0), -1)
        lin_out = self.lin(cv_out)
        return lin_out

class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        lin_num_neurons=[128, 256, 512, 512*3*3],
        lin_activation=nn.ReLU(),
        deconv_activation=nn.ReLU(),
        deconv_kernel_size=(3, 3),
        deconv_stride=2,
        deconv_padding=(1, 1),
        deconv_filters=[256, 128, 64, 32, 16, 8],
        output_shape=(128, 128),
    ):
        super(Decoder, self).__init__()
                
        lin_layers = [
            nn.Linear(
                    in_features=latent_dim, 
                    out_features=lin_num_neurons[0],
                ),
                lin_activation,
        ]
        
        for i in range(len(lin_num_neurons) - 1):
            lin_layers.extend([
                nn.Linear(
                    in_features=lin_num_neurons[i], 
                    out_features=lin_num_neurons[i + 1],
                ),
                lin_activation,
            ])
        
        self.lin_dec = nn.Sequential(*lin_layers)
        
        self.initial_channels = lin_num_neurons[-1] // (3 * 3) 
        self.initial_height = 3
        self.initial_width = 3
        
        deconv_layers = [
            nn.ConvTranspose2d(
                in_channels=self.initial_channels, 
                out_channels=deconv_filters[0],
                kernel_size=deconv_kernel_size,
                stride=deconv_stride,
                padding=deconv_padding,
                ),
            nn.BatchNorm2d(deconv_filters[0]),
            deconv_activation,
        ]
        
        for i in range(len(deconv_filters)):
            if i == (len(deconv_filters) - 1):
                deconv_layers.extend([
                    nn.ConvTranspose2d(
                        in_channels=deconv_filters[-1], 
                        out_channels=1,
                        kernel_size=deconv_kernel_size,
                        stride=deconv_stride,
                        padding=deconv_padding,
                    ),
                    nn.Sigmoid(),
                ])
                
            else:
                deconv_layers.extend([
                    nn.ConvTranspose2d(
                        in_channels=deconv_filters[i], 
                        out_channels=deconv_filters[i + 1],
                        kernel_size=deconv_kernel_size,
                        stride=deconv_stride,
                        padding=deconv_padding,
                    ),
                    nn.BatchNorm2d(deconv_filters[i + 1]),
                    deconv_activation,
                ])
                                
        self.deconv_dec = nn.Sequential(*deconv_layers)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=output_shape)
        
    def forward(self, x):
        lin_out = self.lin_dec(x)
        deconv_in = lin_out.reshape(-1, self.initial_channels, self.initial_height, self.initial_width)
        deconv_out = self.deconv_dec(deconv_in)
        return self.adaptive_pool(deconv_out)
    
class AE(nn.Module):
    def __init__(
        self,
        device='cpu',
        latent_dim=64,
        lin_activation=nn.ReLU(),
        cv_activation=nn.ReLU(),
        lin_num_neurons_decoder=[128, 256, 512, 512*3*3],
        deconv_kernel_size_decoder=(3, 3),
        deconv_padding_decoder=(1, 1),
        deconv_stride_decoder=2,
        deconv_filters_decoder=[256, 128, 64, 32, 16, 8],
        cv_channels_encoder=[1, 16, 32, 64, 128, 256, 512],
        cv_kernel_encoder=(3, 3),
        cv_stride_encoder=2,
        cv_padding_encoder=(1, 1),
        lin_num_neurons_encoder=[256, 128],
        shape=(128, 128),
    ):
        super(AE, self).__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        
        self.encoder = Encoder(
            latent_dim=latent_dim,
            cv_channels=cv_channels_encoder,
            cv_kernel=cv_kernel_encoder,
            cv_stride=cv_stride_encoder,
            cv_padding=cv_padding_encoder,
            cv_activation=cv_activation,
            lin_num_neurons=lin_num_neurons_encoder,
            lin_activation=lin_activation,
            input_shape=shape,
        )
        
        self.decoder = Decoder(
            latent_dim=latent_dim,
            lin_num_neurons=lin_num_neurons_decoder,
            lin_activation=lin_activation,
            deconv_activation=cv_activation,
            deconv_kernel_size=deconv_kernel_size_decoder,
            deconv_stride=deconv_stride_decoder,
            deconv_padding=deconv_padding_decoder,
            deconv_filters=deconv_filters_decoder,
            output_shape=shape,
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)
        return dec_out
    
    def fit(
        self, 
        loader_train,
        loader_val,
        optimizer, 
        criterion, 
        epochs=100, 
        model_path='best_models/autoencoder.pth',
        patience=20,  
        verbose=True,
    ):
        self.to(self.device)

        losses_train, losses_val = [], []
        best_loss = float('inf')
        counter = 0  

        for epoch in range(epochs):
            self.train()  
            loss_train_epoch = 0.0

            for x_batch, y_batch in loader_train:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()  
                y_pred = self.forward(x_batch)  
                loss = criterion(y_pred, y_batch)  
                loss.backward() 
                optimizer.step() 

                loss_train_epoch += loss.item()

            loss_train_epoch /= len(loader_train)
            losses_train.append(loss_train_epoch)

            self.eval()
            loss_val_epoch = 0.0
            with torch.no_grad():
                for x_batch, y_batch in loader_val:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.forward(x_batch)  
                    loss = criterion(y_pred, y_batch)  
                    loss_val_epoch += loss.item()

                loss_val_epoch /= len(loader_val)
                losses_val.append(loss_val_epoch)

            if verbose:
                print(f'Epoch {epoch+1}/{epochs} | Train_loss: {loss_train_epoch:.6f} | Val_loss: {loss_val_epoch:.6f}')

            if loss_val_epoch < best_loss:
                best_loss = loss_val_epoch  
                counter = 0  
                torch.save(self.state_dict(), model_path)  
            else:
                counter += 1

            if counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1}.')
                break

        self.losses_train = losses_train
        self.losses_val = losses_val
        self.model_path = model_path
