import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from hyptorch.nn import ToPoincare

sys.path.append(os.path.dirname(os.getcwd()))
import torch.nn as nn
from utils import euclidean_metric
from torchdiffeq import odeint as odeint
from torch.distributions.multivariate_normal import MultivariateNormal

from models.transform import MultiHeadAttention
from controller.controller import Controller
from hyptorch.pmath import logmap0, expmap0, expmap, mobius_add
from controller.controller_021 import Controller021
from hyptorch.pmath import dist_matrix

from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval
from models.HFA import HFALoss

mean_var_path='/media/mcislab/gaozhi/disk_1/augmentation_ODE/inductive_oneshot_allL_8.23_compute_meanvar/mean_var/100.npy'

EPS = {torch.float32: 1e-8, torch.float64: 1e-8}


def transp(x, u, k):
    return (2/lambda_x(x, k, keepdim=True))*u

def lambda_x( x, k, keepdim=False):
    denominator=(1 + k * (x.norm(dim=-1, p=2, keepdim=keepdim)).pow(2))
    repalce=(torch.ones(denominator.shape).cuda())*EPS[x.dtype]
    denominator=torch.where(denominator==0,repalce,denominator)

    return 2 / denominator
    


class Difference_Estimator (torch.nn.Module):
    def __init__(self, dim, shot ):
        super(Difference_Estimator,self).__init__()

        self.dim=dim
        self.shot=shot


    def forward(self, input_x_mean, input_x):

        #input_x_mean(class,dim)
        #input_x(class_num, self.shot, dim)

        class_num=input_x.shape[0]
        proceed_input_x_class=input_x.view(class_num, self.shot, self.dim)

        same_class_difference = torch.zeros(class_num, self.shot, self.dim).cuda()
        different_class_difference = torch.zeros(class_num, self.shot*(class_num-1), self.dim).cuda()

        extend1_input_x_mean=input_x_mean.repeat(1,self.shot).view(class_num, self.shot, self.dim)
        same_class_difference=proceed_input_x_class-extend1_input_x_mean



        for i in range(class_num):
            different_class_difference[i,:,:]=extend1_input_x_mean[i,:,:].repeat((class_num-1),1) - torch.cat([proceed_input_x_class[0:i,:,:].reshape((i)*self.shot, self.dim), proceed_input_x_class[i+1:class_num,:,:].reshape((class_num-1-i)*self.shot, self.dim) ], dim=0)    

        return same_class_difference, different_class_difference



class Mean_Aggregator (torch.nn.Module): 
    def __init__(self, dim, shot ):
        super(Mean_Aggregator,self).__init__()

        self.dim=dim
        self.shot=shot  

        self.embedding_same = nn.Linear(self.dim, self.dim)
        self.embedding_diff = nn.Linear(self.dim, self.dim)
        self.Interaction = MultiHeadAttention(1, self.dim, self.dim, self.dim, dropout=0.5)  
        self.output = nn.Linear (self.dim,1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t=None, x=None):

        weight_difference, same_class_difference, different_class_difference =x
        class_num=weight_difference.shape[0]

        weight_difference=weight_difference.view(class_num, self.dim) 

        proceed_same_class_difference = self.embedding_same(same_class_difference)
        proceed_different_class_difference = self.embedding_diff(different_class_difference)

        proceed_same_class_difference=torch.transpose(proceed_same_class_difference,0,1)
        proceed_different_class_difference=torch.transpose(proceed_different_class_difference,0,1)

        proceed_same_class_difference_difference=torch.transpose(proceed_same_class_difference-weight_difference,0,1)
        proceed_different_class_difference_difference=torch.transpose(proceed_different_class_difference-weight_difference,0,1)

        transformer_same=self.Interaction(proceed_same_class_difference_difference,proceed_same_class_difference_difference,proceed_same_class_difference_difference)
        transformer_different=self.Interaction(proceed_different_class_difference_difference,proceed_different_class_difference_difference,proceed_different_class_difference_difference)


        output_same=self.output(transformer_same)
        output_different=self.output(transformer_different)

        softmax_same=self.softmax(output_same)  #(self.class_num, self.shot, 1)
        softmax_different=self.softmax(output_different)  #(self.class_num, self.shot*(self.class_num-1), 1)

        weight_same=softmax_same*proceed_same_class_difference_difference
        weight_difference=softmax_different*proceed_different_class_difference_difference

        weight_same=weight_same.view(class_num, self.shot, self.dim)
        weight_difference=weight_difference.view(class_num, self.shot*(class_num-1), self.dim)

        weight_difference=torch.cat([weight_same,weight_difference],dim=1)  ##(self.class_num, self.shot, self.shot*self.class_num, self.dim)

        weight_difference = torch.mean(weight_difference,dim=1)

        return (weight_difference,torch.zeros(class_num, self.shot, self.dim).cuda(), torch.zeros(class_num, self.shot*(class_num-1), self.dim).cuda())







class Cov_Aggregator (torch.nn.Module): 
    def __init__(self, dim, shot ):
        super(Cov_Aggregator,self).__init__()

        self.dim=dim
        self.shot=shot  

        self.embedding_L = nn.Linear(self.dim, self.dim)

        self.Interaction = MultiHeadAttention(1, self.dim, self.dim, self.dim, dropout=0.5)  
 
        self.output = nn.Linear (self.dim, self.dim)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, t=None, x=None):

        L, same_class_difference, different_class_difference = x
        class_num=L.shape[0]
        ## L (self.class_num, self.dim, self.dim)
        ## same_class_difference (self.class_num, self.shot, self.dim)
        ## different_class_difference (self.class_num, self.shot*(self.class_num-1), self.dim)

        L_projection=self.embedding_L(L.view(class_num*self.dim, self.dim))
        L=L.view(class_num, self.dim, self.dim)
        L_projection=L_projection.view(class_num, self.dim, self.dim)

        L_mean = torch.mean(L, dim=1)

        L_mean=L_mean.view(class_num, self.dim)

        same_class_difference = same_class_difference.view(class_num, self.shot, self.dim)
        different_class_difference = different_class_difference.view(class_num, self.shot*(class_num-1), self.dim)      

        proceed_same_class_difference=torch.transpose(same_class_difference,0,1)
        proceed_different_class_difference=torch.transpose(different_class_difference,0,1)

        proceed_same_class_difference_difference=torch.transpose(L_mean-proceed_same_class_difference,0,1)
        proceed_different_class_difference_difference=torch.transpose(L_mean-proceed_different_class_difference,0,1)

        proceed_class_difference_difference = torch.cat([proceed_same_class_difference_difference,proceed_different_class_difference_difference], dim=1)

        transformer=self.Interaction(L-L_projection, proceed_class_difference_difference, proceed_class_difference_difference)


        L_grad=self.output(transformer)


        return (L_grad, torch.zeros(class_num, self.shot, self.dim).cuda(), torch.zeros(class_num, self.shot*(class_num-1), self.dim).cuda())






class Curvature_Aggregator(torch.nn.Module):
    def __init__(self, dim, shot):
        super(Curvature_Aggregator,self).__init__()

        self.dim=dim
        self.shot=shot

        self.embedding1 = nn.Linear(self.dim+1, self.dim+1)
        self.embedding2 = nn.Linear(self.dim+1, 1)

        self.Interaction = MultiHeadAttention(1, self.dim+1, self.dim+1, self.dim+1, dropout=0.5)

    def forward(self, t=None, x=None):

        # k(class_num)
        # data (class_num, self.dim)
        curvature, data = x
        class_num=curvature.shape[0]

        curvature=curvature.reshape(class_num,1)

        togetherdata=torch.cat([data,curvature],dim=1)

        togetherdata=self.embedding1(togetherdata)
        togetherdata = self.Interaction(togetherdata.unsqueeze(0), togetherdata.unsqueeze(0), togetherdata.unsqueeze(0))

        togetherdata=self.embedding2(togetherdata)

        return (togetherdata.squeeze(), torch.zeros(class_num, self.dim).cuda())


class Cov_Decomposer (torch.nn.Module): 
    def __init__(self, dim, shot ):
        super(Cov_Decomposer,self).__init__()

        self.dim=dim
        self.shot=shot  

        self.Embedding = MultiHeadAttention(1, self.dim, self.dim, self.dim, dropout=0.5)
        self.stdcov=nn.Parameter(torch.randn(self.dim,self. dim), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, weight_difference):

        class_num=weight_difference.shape[0]
        weight_difference=weight_difference.view(class_num, self.shot*class_num, self.dim)
        weight_difference=self.Embedding(weight_difference,weight_difference,weight_difference)
        weight_difference=weight_difference.view(class_num, self.shot*class_num, self.dim)


        final_weight = torch.bmm(weight_difference.view(class_num, self.shot*class_num, self.dim), self.stdcov.repeat(class_num,1,1) )  #(self.class_num, self.shot*self.class_num, dim)
        final_weight =  self.softmax(final_weight)
        final_stdcov = torch.bmm(torch.transpose(weight_difference.view(class_num, self.shot*class_num, self.dim) ,1,2 ), final_weight)


        return final_stdcov.view(class_num, self.dim, self.dim)







class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x[0])
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-2, atol=1e-2, method='rk4')
        return out[0][1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value






class Augment_weight(nn.Module):
    def __init__(self, dim, rho, r):
        super(Augment_weight, self).__init__()

        self.dim=dim
        self.rho=rho
        self.r=r

        self.Embedding = nn.Linear(self.dim, self.dim)
        self.Interaction = MultiHeadAttention(1, self.dim, self.dim, self.dim, dropout=0.5)  
        self.Projection = nn.Linear(self.dim, 1)
        self.softmax = nn.Softmax(dim=1)

        self.all_weight = Controller021(self.dim, rho, rho, r)

    def forward(self,non_Euclidean_data,augmented_data):

        #non_Euclidean_x(class_num*shot, dim)
        #augmented_data(class_num*shot*sample_num, dim)

        proceed_data=self.Embedding(augmented_data)
        transformer_data=self.Interaction(proceed_data.unsqueeze(0), proceed_data.unsqueeze(0), proceed_data.unsqueeze(0))
        Projection_data=self.Projection(transformer_data)

        augmentation_weight=self.softmax(Projection_data)*augmented_data.shape[0]
        augmentation_weight=augmentation_weight.view(-1)

        all_weight=self.all_weight(augmented_data,non_Euclidean_data)

        return augmentation_weight, all_weight





class ProtoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
                            
        self.shot = self.args.shot
        self.train_way = self.args.train_way
        self.validation_way=self.args.validation_way
        self.curvaturedim=self.args.curvaturedim
        self.dim=self.args.dim

        self.sample_num = self.args.sample_num
        self.augment_lambda=self.args.augment_lambda

        self.rho=self.args.rho
        self.r=self.args.r

        self.temperature=self.args.temperature
        self.train_step=self.args.train_step

        self.query_num=self.args.query_num
        self.innerlr=self.args.innerlr


        self.de = Difference_Estimator (self.curvaturedim, self.shot)
        self.ma = Mean_Aggregator(self.curvaturedim, self.shot)
        self.ca = Cov_Aggregator(self.curvaturedim, self.shot)
        self.cura=Curvature_Aggregator(self.curvaturedim, self.shot)
        self.cd = Cov_Decomposer(self.curvaturedim, self.shot)

        self.model_initialzation(self.de)
        self.model_initialzation(self.ma)
        self.model_initialzation(self.ca)
        self.model_initialzation(self.cd)

        self.Mean_ODE=ODEBlock(self.ma)
        self.Cov_ODE=ODEBlock(self.ca)
        self.Cur_ODE=ODEBlock(self.cura)

        self.controller = Controller( self.curvaturedim, self.rho, self.rho, self.r)

        self.model_initialzation(self.controller)


        mean_var=np.load(mean_var_path,allow_pickle=True)

        self.mean_mean=nn.Parameter(torch.Tensor(mean_var[8]))
        self.mean_var=nn.Parameter(torch.Tensor(mean_var[9]))
        self.var_mean=nn.Parameter(torch.Tensor(mean_var[10]))
        self.var_var=nn.Parameter(torch.Tensor(mean_var[11]))

        #self.tangentpoint=nn.Parameter(torch.randn(1,self.args.dim))

        self.lambda_0=nn.Parameter(torch.ones(1,1)*self.args.augment_lambda)
        self.lambda_1=nn.Parameter(torch.ones(1,1)*self.args.augment_lambda)

        self.dimensionreduction = nn.Linear(self.args.dim, self.curvaturedim)


        self.proto_attn = MultiHeadAttention(5, self.args.dim, self.args.dim, self.args.dim, dropout=0.5)  
        self.innerlr = nn.Parameter(torch.sqrt(torch.ones(self.args.train_step)*self.args.innerlr))
        self.criterion = HFALoss()





    def model_initialzation(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)


    def lambda_x( self, x, k, keepdim=False):
        denominator=(1 + k * (x.norm(dim=-1, p=2, keepdim=keepdim)).pow(2))
        repalce=(torch.ones(denominator.shape).cuda())*EPS[x.dtype]
        denominator=torch.where(denominator==0,repalce,denominator)

        return 2 / denominator

    def parallel_transport(self, x_t_1, x_t, old_g, k):

        return transp(x_t_1, x_t, old_g, k)

    def orthogonality(self,x,Eu_grad,curvature):

        lambda_x=self.lambda_x(x,curvature,True)
        return (1/(lambda_x*lambda_x))*Eu_grad



    def get_per_step_loss_importance_vector(self,current_epoch):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.train_step)) * (
                1.0 / self.args.train_step)
        decay_rate = 1.0 / self.args.train_step / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.train_step
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (current_epoch * (self.args.train_step - 1) * decay_rate),
            1.0 - ((self.args.train_step - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).cuda()
        return loss_weights
        


    def forward(self, shot, query, epoch, query_label):

        if self.training:
            self.query_num=self.args.query_num
        else:
            self.query_num=15

        data_shot=shot
        data_query=query

        if self.training:
            class_num=self.train_way
        else:
            class_num=self.validation_way
        
        data_shot_category = torch.transpose(data_shot.reshape(self.shot, class_num, -1), 0,1)  #size is (class_num, self.shot, -1)
        data_query_category=torch.transpose(data_query.reshape(self.query_num, class_num, -1), 0,1) #size is (class_num, self.query_num, -1)

        data_shot_category_mean=torch.mean(data_shot_category,dim=1)

        trans_proto=self.proto_attn(data_shot_category_mean.unsqueeze(0),data_shot_category_mean.unsqueeze(0),data_shot_category_mean.unsqueeze(0))
        trans_proto_norm=torch.norm(trans_proto,dim=-1, keepdim=True)
        trans_proto=trans_proto/(trans_proto_norm)


        #-------------------1.measure mean and covariance------------------------ 
        same_class_difference, different_class_difference = self.de(data_shot_category_mean,data_shot_category)  #different_class_difference(class_num, (class_num-1), self.dim)
        different_class_difference=different_class_difference/2
        difference=torch.cat([same_class_difference,different_class_difference], dim =1)
        mean=torch.mean(difference, dim=1)
        mean_difference=torch.mean(different_class_difference, dim=1)
        mean_distance=torch.sum(mean_difference*mean_difference)
        L=self.cd(difference)

        final_mean=self.Mean_ODE(x=(mean,same_class_difference,different_class_difference))#*(epoch/self.args.max_epoch)
        final_cov=self.Cov_ODE(x=(L,same_class_difference,different_class_difference))#*(epoch/self.args.max_epoch)
        final_cov=final_cov.reshape(class_num, self.curvaturedim, self.curvaturedim)




        ## -------------------2.differentiable sample vectors from batch of distributions-------------------
        final_mean=final_mean.reshape(class_num, self.curvaturedim) ##(class_num, self.dim)
        final_cov=(final_cov.reshape(class_num, self.curvaturedim, self.curvaturedim)) ##(class_num, self.dim,  self.dim)

        m = MultivariateNormal(torch.zeros(self.curvaturedim).cuda(), torch.eye(self.curvaturedim).cuda())
        sampled_vectors=m.sample([class_num*self.sample_num])

        sampled_vectors=sampled_vectors.reshape(self.sample_num, class_num, self.curvaturedim)
        sampled_vectors=sampled_vectors+final_mean
        sampled_vectors=torch.bmm(final_cov.repeat(self.sample_num,1,1).reshape(self.sample_num*class_num, self.curvaturedim, self.curvaturedim), sampled_vectors.reshape(self.sample_num*class_num, self.curvaturedim,1))
        sampled_vectors=sampled_vectors.reshape(self.sample_num, class_num, self.curvaturedim)

        sampled_vectors=sampled_vectors.reshape(self.sample_num*class_num, self.curvaturedim)
        sampled_vectors=sampled_vectors.reshape(self.sample_num, class_num, self.curvaturedim)
        sampled_vectors=sampled_vectors
        





        ## -------------------3.generated curvature-------------------
        curvature = self.controller(data_shot_category_mean, data_shot_category_mean)  #(class_num)
        curvature=self.Cur_ODE(x=(curvature,data_shot_category_mean))
        curvature=(F.sigmoid(curvature*self.args.curvaturel))*self.args.curvaturescale+self.args.curvaturestart
        if self.args.curvature!=0:
        	curvature=torch.ones(class_num)*self.args.curvature



        
        ## -------------------4.generated some augmented samples-------------------
        #generate some data on constant curvature spaces
        augmented_data_zerotangent=torch.zeros(class_num, self.sample_num, self.curvaturedim).cuda()
        augmented_data=torch.zeros(class_num, self.sample_num, self.curvaturedim).cuda()         
        for i in range(class_num):        

            augmented_data_zerotangent_i=(sampled_vectors[:,i,:]+trans_proto.squeeze()[i,:].unsqueeze(0))
            augmented_data_zerotangent_i_norm=torch.norm(augmented_data_zerotangent_i,dim=-1, keepdim=True)
            augmented_data_zerotangent_i=augmented_data_zerotangent_i/augmented_data_zerotangent_i_norm                
            augmented_data_zerotangent[i,:,:]=augmented_data_zerotangent_i
            augmented_data[i,:,:]=expmap0(augmented_data_zerotangent_i, k=curvature[i])




        ## -------------------5.generate classifier-------------------
        classifier=nn.Parameter(trans_proto.squeeze(), requires_grad=True)



        ## -------------------8.various property-------------------
        dis_proto=torch.zeros(class_num,class_num).cuda()
        for i in range(class_num):
            dis_proto[i,:]= (dist_matrix(expmap0(trans_proto.squeeze(), k=curvature[i]), expmap0(trans_proto.squeeze()[i,:].unsqueeze(0), k=curvature[i]), k=curvature[i])).squeeze()
        mean_distance_rie=torch.sum(dis_proto)/(class_num*class_num-class_num)
        mean_distance_rie=mean_distance_rie/2

        if self.training:
            augmented_data_zerotangent=augmented_data_zerotangent.reshape(class_num, self.sample_num, self.curvaturedim)
            augmented_prototype_distance=torch.zeros(class_num, self.sample_num).cuda()
            for i in range(class_num):
                augmented_prototype_distance[i,:]= (dist_matrix(augmented_data[i,:,:], expmap0(trans_proto.detach().squeeze()[i].unsqueeze(0), k=curvature[i]), k=curvature[i])).squeeze()
            augmented_prototype_distance=torch.where(augmented_prototype_distance<mean_distance_rie, mean_distance_rie-augmented_prototype_distance, torch.zeros(augmented_prototype_distance.shape).cuda())
            loss_diversity1=torch.mean(augmented_prototype_distance.pow(2))
            

        if self.training:
            augmented_distance=torch.zeros(class_num, self.sample_num, self.sample_num).cuda()
            for i in range (class_num):
                augmented_distance[i,:,:]=dist_matrix(augmented_data[i,:,:], augmented_data[i,:,:], k=curvature[i])
            augmented_distance=torch.where(augmented_distance<mean_distance_rie, mean_distance_rie-augmented_distance, torch.zeros(augmented_distance.shape).cuda())
            loss_diversity2=torch.mean(augmented_distance.pow(2))
            loss_diversity=loss_diversity1+loss_diversity2
        



        # -------------------6.classification in training manifold data-------------------

        transap_classifier=torch.zeros(trans_proto.squeeze().shape).cuda()

        save_loss=torch.zeros(self.train_step)
        save_acc=torch.zeros(self.train_step)
        query_acc=torch.zeros(self.train_step+1)

        loss_meta=0
        per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector(epoch)


        data_query_category=data_query_category.reshape(class_num*self.query_num, self.curvaturedim)
        query_logit=torch.zeros((self.query_num)*class_num, class_num).cuda()
        for j in range(class_num):
            query_logit[:,j] = - (dist_matrix(expmap0(data_query_category, k=curvature[j]), expmap0(classifier[j].unsqueeze(0), k=curvature[j]), k=curvature[j])/self.temperature).squeeze()

        query_logit = query_logit.reshape(class_num, self.query_num, class_num)
        query_logit = torch.transpose(query_logit,0,1)
        query_logit = query_logit.reshape(self.query_num*class_num, class_num)

        loss_q=F.cross_entropy(query_logit, query_label)
        acc_q=count_acc(query_logit, query_label)
        query_acc[0]=acc_q   


        for i in range(self.train_step):

            aug_mean=final_mean*(self.lambda_1[0,0]*self.lambda_1[0,0])+trans_proto.squeeze()

            for j in range(class_num):
                transap_classifier[j]=expmap0(classifier[j].unsqueeze(0), k=curvature[j])

            y = torch.mm(aug_mean, transap_classifier.t())

            target_x = torch.arange(class_num)
            target_x = target_x.type(torch.cuda.LongTensor)

            loss,_=self.criterion(transap_classifier, y, aug_mean, target_x, self.lambda_0[0,0]*self.lambda_0[0,0], final_cov, class_num)
            save_loss[i]=loss.detach()

            if self.training:
                classifier_grad = torch.autograd.grad(loss, classifier, retain_graph=True, allow_unused=True)
            else:
                classifier_grad = torch.autograd.grad(loss, classifier)
            classifier =classifier-(self.innerlr[i]*self.innerlr[i])*classifier_grad[0]


            data_query_category=data_query_category.reshape(class_num*self.query_num, self.curvaturedim)
            query_logit=torch.zeros((self.query_num)*class_num, class_num).cuda()
            for j in range(class_num):
                query_logit[:,j] = - (dist_matrix(expmap0(data_query_category, k=curvature[j]), expmap0(classifier[j].unsqueeze(0), k=curvature[j]), k=curvature[j])/self.temperature).squeeze()

            query_logit = query_logit.reshape(class_num, self.query_num, class_num)
            query_logit = torch.transpose(query_logit,0,1)
            query_logit = query_logit.reshape(self.query_num*class_num, class_num)

            loss_q=F.cross_entropy(query_logit, query_label)
            acc_q=count_acc(query_logit, query_label)
            query_acc[i+1]=acc_q   

            loss_meta=loss_meta+per_step_loss_importance_vectors[i]*loss_q


        save_loss=save_loss.numpy().tolist()





        if self.training:
            augmented_data_zerotangent=augmented_data_zerotangent.reshape(class_num, self.sample_num, self.dim)
            data_query_category=data_query_category.reshape(class_num*self.query_num, self.dim)
            query_logit_sample=torch.zeros(self.sample_num, class_num*(self.query_num), class_num).cuda()
            for j in range(class_num):
                for jj in range(self.sample_num):
                    query_logit_sample[jj,:,j] = - (dist_matrix(expmap0(data_query_category, k=curvature[j]), expmap0(augmented_data_zerotangent[j,jj,:].unsqueeze(0), k=curvature[j]), k=curvature[j])/self.temperature).squeeze()

            query_logit_sample = query_logit_sample.reshape(self.sample_num, class_num, self.query_num, class_num)
            query_logit_sample = torch.transpose(query_logit_sample,1,2)
            query_logit_sample = query_logit_sample.reshape(self.sample_num, self.query_num*class_num, class_num)




        data_query_category=data_query_category.reshape(class_num*self.query_num, self.curvaturedim)
        query_logit=torch.zeros((self.query_num)*class_num, class_num).cuda()
        for j in range(class_num):
            query_logit[:,j] = - (dist_matrix(expmap0(data_query_category, k=curvature[j]), expmap0(classifier[j].unsqueeze(0), k=curvature[j]), k=curvature[j])/self.temperature).squeeze()

        query_logit = query_logit.reshape(class_num, self.query_num, class_num)
        query_logit = torch.transpose(query_logit,0,1)
        query_logit = query_logit.reshape(self.query_num*class_num, class_num)


        if self.training:
            return query_logit, query_logit_sample, loss_diversity, 0, loss_meta
        else:
            return query_logit, query_acc
