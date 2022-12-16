import torch
import torch.nn as nn
from torch.autograd import Variable as V
from controller.MatrixBiMul import MatrixBiMul
import torch.nn.functional as F


class Controller021(torch.nn.Module):
    
    def __init__(self, backbone_input_dim,hidden_dim,output_dim,factor_num):
        super(Controller021,self).__init__()
                
        self.l=0.1

        self.backbone_input_dim=backbone_input_dim
        self.output_dim=output_dim
        self.hidden_dim=hidden_dim
        self.factor_num=factor_num

        
        self.proto_linear = nn.Linear(backbone_input_dim,hidden_dim*factor_num, bias=False)
        self.all_linear = nn.Linear(backbone_input_dim,hidden_dim*factor_num, bias=False)

        #self.fclayer=nn.Linear(hidden_dim,output_dim)
        self.predictor1 = nn.Linear(hidden_dim,hidden_dim)
        self.predictor2 = nn.Linear(hidden_dim,hidden_dim)
        self.predictor3 = nn.Linear(hidden_dim,1)
        

        nn.init.xavier_normal(self.proto_linear.weight)
        nn.init.xavier_normal(self.all_linear.weight)


    def forward(self, mean_proto_category, all_data):
        
        proto_data=self.proto_linear(mean_proto_category)
        all_data=self.all_linear(all_data)
        c = torch.mul(proto_data, all_data)

        
        c = c.view(-1, self.hidden_dim, self.factor_num)
        c = torch.sum(c, 2)

        c = torch.sum(c, dim=0, keepdim=True)

        c = torch.sqrt(F.relu(c)) - torch.sqrt(F.relu(-c))  
        c = F.normalize(c, p=2, dim=1) 

        #c = F.relu(c,inplace=True)
        #c = self.fclayer(c)
        #c = F.relu(c)
        #c = self.fclayer2(c)
        #c = F.relu(c)
        

        '''
        c = c.view(-1, self.output_dim, self.factor_num)
        c = torch.squeeze(torch.sum(c, 2))
        c = F.relu(c)
        '''

        output9 = self.predictor1(c)
        output9 = F.relu(output9)
        output9 = self.predictor2(output9)
        output9 = F.relu(output9)
        output9 = self.predictor3(output9)        
        #rint('output9',output9)
        output10=torch.randn(output9.shape).cuda()
        output10=F.sigmoid(output9)
        output10=torch.squeeze(output10)

        return output10