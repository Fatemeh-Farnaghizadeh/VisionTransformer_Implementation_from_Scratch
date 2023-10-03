import utils
import torch
import einops

from torch import nn


##############################################################
class PatchEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        assert utils.IMG_SIZE % utils.PATCH_SIZE == 0, "image_size should be divisible by patch_size"

        self.num_paches = (
            (utils.IMG_SIZE // utils.PATCH_SIZE) * (utils.IMG_SIZE // utils.PATCH_SIZE)
        )

        self.flatt_input_dim = (
            utils.PATCH_SIZE * utils.PATCH_SIZE * utils.IMG_CHANNELS
        )  

        self.linearProjection = nn.Linear(self.flatt_input_dim, utils.EMBED_DIM)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, utils.EMBED_DIM),
            requires_grad=True
        ).to(utils.DEVICE)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_paches+1, utils.EMBED_DIM),
            requires_grad=True
        ).to(utils.DEVICE)

    def forward(self, x):

        pached_imgs = einops.rearrange(
            x, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)',
            h1=utils.PATCH_SIZE, w1=utils.PATCH_SIZE
        )

        linear_projection = self.linearProjection(pached_imgs)
        b, n_p, _ = linear_projection.shape

        #cls_token-->(b, 1, embed_dim)
        self.cls_token_embedded = self.cls_token.repeat(b, 1, 1)

        #linear_projection --> (b, n_patch+1, embed_dim)
        linear_projection = torch.cat(
            (self.cls_token_embedded, linear_projection), dim=1
        )

        #broadcasting
        linear_projection += self.pos_embedding

        return linear_projection
    
    
##############################################################

class EncoderBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.first_norm = nn.LayerNorm(utils.EMBED_DIM)
        self.MHA = nn.MultiheadAttention(
            utils.EMBED_DIM,
            utils.NUM_HEADS,
            utils.ATT_DROP,
            batch_first=True
        )

        self.MLP = nn.Sequential(
            nn.Linear(utils.EMBED_DIM, utils.EMBED_DIM*4),
            nn.GELU(),
            nn.Dropout(utils.MLP_DROP),
            nn.Linear(utils.EMBED_DIM*4, utils.EMBED_DIM),
            nn.Dropout(utils.MLP_DROP)
        )
        
        self.secound_norm = nn.LayerNorm(utils.EMBED_DIM)

    def forward(self, pach_embedding):
        first_norm_out = self.first_norm(pach_embedding)
        MHA_out = self.MHA(first_norm_out, first_norm_out, first_norm_out)[0]

        first_added = MHA_out + pach_embedding
        
        secound_norm_out = self.secound_norm(first_added)
        mlp_out = self.MLP(secound_norm_out)

        final_out = first_added + mlp_out

        return final_out
    

##############################################################

class ViT(nn.Module):

    def __init__(self):
        super(ViT, self).__init__()

        self.embedding = PatchEmbedding()

        # Create the stack of encoders
        self.encStack = nn.ModuleList([
            EncoderBlock() for i in range(utils.NUM_ENCODERS)
        ])

        self.MLP_head = nn.Sequential(
            nn.LayerNorm(utils.EMBED_DIM),
            nn.Linear(utils.EMBED_DIM, utils.EMBED_DIM),
            nn.Linear(utils.EMBED_DIM, utils.NUM_CLASSES)
        )

    def forward(self, x):
        enc_output = self.embedding(x)

        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        cls_token_embed = enc_output[:, 0]

        return self.MLP_head(cls_token_embed)
    
    
##############################################################


if __name__ == '__main__':
    ## Test PatcEmbedding

    # projection_model = PatchEmbedding().to(utils.DEVICE)
    # x = torch.randn(8, 3, 224, 224).to(utils.DEVICE)
    # res = projection_model(x)

    # #Test EncoderBlock
    # encoder_block = EncoderBlock().to(utils.DEVICE) 
    # x = torch.randn(8, 197, 32).to(utils.DEVICE)
    # res = encoder_block(x)

    #Test ViT
    encoder_block = ViT().to(utils.DEVICE) 
    x = torch.randn(16, 3, 224, 224).to(utils.DEVICE)
    res = encoder_block(x)
    
    print(res.shape)
    print(res)