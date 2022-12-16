import torch
import torch.nn as nn



class HFALoss(nn.Module):
    def __init__(self):
        super(HFALoss, self).__init__()


        self.cross_entropy = nn.CrossEntropyLoss()

    def hfa_aug(self, weight_m, features, y, labels, ratio, final_conv, class_num):

        N = features.size(0)
        C = class_num
        A = features.size(1)

        #print('weight_m',weight_m.shape)

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = final_conv

        sigma1=torch.sum(NxW_ij*NxW_ij,dim=-1)-torch.sum(NxW_kj*NxW_kj,dim=-1)

        #print('CV_temp',CV_temp.shape)

        #sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)

        sigma2 = ratio * \
                  torch.bmm(torch.bmm(NxW_ij - NxW_kj,
                                      CV_temp),
                            (NxW_ij - NxW_kj).permute(0, 2, 1))
        #print('sigma2',sigma2.shape)
        sigma2 = sigma2.mul(torch.eye(C).cuda()
                             .expand(N, C, C)).sum(2).view(N, C)
        #print('sigma2',sigma2.shape)

        #sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
        #    CV_temp.view(N, 1, A).expand(N, C, A)
        #).sum(2)
        aug_result = y + 0.5 * sigma2 + sigma1

        return aug_result

    def forward(self, weight_m, y, features, target_x, ratio, final_conv, class_num):

        # y = fc(features)

        hfa_aug_y = self.hfa_aug(weight_m, features, y, target_x, ratio, final_conv, class_num)
        #hfa_aug_y=hfa_aug_y/1000
        #print('hfa_aug_y',hfa_aug_y)

        loss = self.cross_entropy(hfa_aug_y, target_x)

        return loss, y


'''
batch_size=16
feature_dim=256
num_classes=10
feature_num=1
ratio=1

feature=torch.randn(num_classes,feature_dim).cuda()

weight_m=nn.Parameter(torch.randn(num_classes,feature_dim), requires_grad=True)
weight_m=weight_m.cuda()

criterion=HFALoss(num_classes).cuda()

target_x = torch.arange(num_classes)
target_x = target_x.type(torch.cuda.LongTensor)

y = torch.mm(feature, weight_m.t())

final_conv=torch.randn(num_classes,feature_dim,feature_dim).cuda()

loss=criterion(weight_m, y, feature, target_x, ratio, final_conv)
'''