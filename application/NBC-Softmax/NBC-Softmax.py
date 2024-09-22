class ProjContrastiveModel_BC1(Module):    
    def __init__(self, proj_dim=16, emb_dim=32, num_classes=-1, tau=0.2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        if proj_dim > 0:
            self.auth_emb = nn.Linear(emb_dim, proj_dim)
        else:
            self.auth_emb = None
        print(f' ProjContrastiveModel_BC1 num_classes:{num_classes} tau:{tau} emb_dim:{emb_dim} proj_dim:{proj_dim} {self.auth_emb}' )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def get_feature_dis(self,x):
        #x :           batch_size x nhid
        #x_dis(i,j):   item means the similarity between x(i) and x(j).
        x_dis = x@x.T
        mask = torch.eye(x_dis.shape[0]).to(x.device)
        x_sum = torch.sum(x**2, 1).reshape(-1, 1)
        x_sum = torch.sqrt(x_sum).reshape(-1, 1)
        x_sum = x_sum @ x_sum.T
        x_dis = x_dis*(x_sum**(-1))
        x_dis = (1-mask) * x_dis
        return x_dis

    def loss_calculation(self, z,c, num_classes, tau):
        #compute cluster centers CC
        ONEHOT = F.one_hot(c, num_classes=num_classes)  # N*num_classes
        CC = torch.sum(z.unsqueeze(1) * ONEHOT.unsqueeze(2) , dim=0) / torch.clamp( ONEHOT.sum(0).unsqueeze(-1), min=1.0)   
        # Z(N,FE)->(N,1,FE) * ONEHOT(N,num_classes)->(N,num_classes,1) ==> (N, num_classes, FE) / (num_classes,1) ==>sum(0) ==> num_classes*FE

        CC_VALID = CC[ ONEHOT.sum(0) > 0 ]
        negative_distances = self.get_feature_dis(CC_VALID)
        negative_distances = torch.exp( tau * negative_distances)
        negative_distances = negative_distances.sum(1)
        lossN = torch.log(negative_distances +1e-10).mean() / num_classes

        return lossN

    def forward(self, embs, labels):
        logits = self.auth_emb(embs) if self.auth_emb is not None else embs
        return self.loss_calculation(logits, labels, self.num_classes, self.tau)

###############################################################################
# Helper function to combine mutiple losses at given ratios
###############################################################################
class NCE_Combiner2(nn.Module): 
    def __init__(self, model1, model2, model2_ratio):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model2_ratio = model2_ratio

    def forward(self, embs, labels):
        loss1 = self.model1(embs, labels)
        loss2 = self.model2(embs, labels)
        return (1-self.model2_ratio)*loss1 + self.model2_ratio*loss2

###############################################################################
# Modified get_classwise_nce_model from https://github.com/pranavmaneriker/SYSML
# Mofifications are to add the NBC-softmax as an auxiliary loss
###############################################################################
def get_classwise_nce_model(len_auth_tok, final_dim, classwise_model_params, hparams):
    if classwise_model_params["model_type"] == "sm":
        sm_model = SoftmaxModel(len_auth_tok, final_dim)
        classwise_model = sm_model
    elif classwise_model_params["model_type"] == "arcface":
        classwise_model = losses.ArcFaceLoss(len_auth_tok, final_dim)
    elif classwise_model_params["model_type"] == "contrastive":
        classwise_model = losses.ContrastiveLoss()
    elif classwise_model_params["model_type"] == "cosface":
        classwise_model = losses.CosFaceLoss(len_auth_tok, final_dim)
    elif classwise_model_params["model_type"] == "infonce":
        classwise_model = losses.NTXentLoss()
    elif classwise_model_params["model_type"] == "ms":
        classwise_model = losses.MultiSimilarityLoss()
    
    elif classwise_model_params["model_type"] == "COMBO2":
        classwise_model_params1 = { "model_type" : classwise_model_params["model1_type"],}
        classwise_model_params2 = { "model_type" : classwise_model_params["model2_type"], "proj_dim" : (int)(classwise_model_params["proj_dim"])}
        model1 = get_classwise_nce_model(len_auth_tok, final_dim, classwise_model_params1, hparams)
        model2 = get_classwise_nce_model(len_auth_tok, final_dim, classwise_model_params2, hparams)
        model2_ratio = (float)(classwise_model_params["model2_ratio"])
        classwise_model = NCE_Combiner2(model1, model2, model2_ratio)
        print(classwise_model)
    else:
        raise NotImplementedError
    return classwise_model