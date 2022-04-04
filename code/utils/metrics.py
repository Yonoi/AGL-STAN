import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from sklearn.metrics import confusion_matrix, classification_report
"""
Metrics 

# Regression Task
1. MAE # as loss function
2. MSE # as loss function
3. RMSE # as test performance
4. MAPE # as test performance

# Binary Classification Task
1. BCEWithLogitsLoss # as loss function
2. Micro-F1 # test performance
3. Macro-F1 # test performance
"""

# for training
class RegressionLoss(nn.Module):
    def __init__(self, scaler, mask_value=1e-5, loss_type='mae'):
        super(RegressionLoss, self).__init__()
        self._scaler = scaler
        self._mask_value = mask_value
        self._loss_type = loss_type
    
    def _inv_transform(self, data):
        return self._scaler.inverse_transform(data)

    def _masked_mae(self, preds, labels):
        return torch.mean(torch.abs(preds - labels))
    
    def _masked_mse(self, preds, labels):
        return torch.mean(torch.square(preds - labels))

    def forward(self, preds, labels):
        # inverse transform
        preds = self._inv_transform(preds)
        labels = self._inv_transform(labels)
        
        if self._mask_value is not None:
            # mask some elements        
            mask = torch.gt(labels, self._mask_value)
            preds = torch.masked_select(preds, mask)
            labels = torch.masked_select(labels, mask)

        if self._loss_type == 'mae':
            return self._masked_mae(preds, labels)
        elif self._loss_type == 'mse':
            return self._masked_mse(preds, labels)
        else:
            raise Exception('Illegal Loss Function\'s Name.')

# for testing or validation
class RegressionMetrics(nn.Module):
    def __init__(self, scaler, mask_value=0.0):
        super(RegressionMetrics, self).__init__()
        self._scaler = scaler
        self._mask_value = mask_value

    def _inv_transform(self, data):
        return self._scaler.inverse_transform(data)

    def _masked_rmse(self, preds, labels):
        return torch.sqrt(torch.mean(torch.square(preds - labels)))

    def _masked_mape(self, preds, labels):
        return torch.mean(torch.abs(torch.div((preds - labels), labels)))

    def forward(self, preds, labels):
        # inverse transform
        preds = self._inv_transform(preds)
        labels = self._inv_transform(labels)

        # mask some elements        
        mask = torch.gt(labels, self._mask_value)
        preds = torch.masked_select(preds, mask)
        labels = torch.masked_select(labels, mask)

        return (
            self._masked_rmse(preds, labels),
            self._masked_mape(preds, labels)
        )

# need to implement for Binary Classification Task
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2, device=torch.device('cuda:0')):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params).to(device)

    def forward(self, x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class ClassificationLoss(nn.Module):
    def __init__(self, pos_weight=None, loss_type='bce', lambda_value=0.5, device=torch.device('cuda:0')):
        super(ClassificationLoss, self).__init__()
        self._pos_weight = pos_weight
        self._lambda_value = lambda_value
        self._loss_type = loss_type
        self._auto_loss = AutomaticWeightedLoss(num=len(pos_weight), device=device)
    
    def _BCEWithLogits(self, pos_weight):
        if pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()

    def _other_loss(self, preds, labels):
        preds = torch.sigmoid(preds)
        numerator = 2 * torch.reshape(preds * labels, (preds.shape[0], -1)).sum(-1)
        denominator = torch.reshape(preds + labels, (preds.shape[0], -1)).sum(-1)
        return torch.mean(1 - numerator / denominator)

    def _main_loss(self, preds, labels, pos_weight):
        if self._loss_type == 'bce':
            loss_fn = self._BCEWithLogits(pos_weight)
            return loss_fn(preds, labels)
        else:
            # maybe i can add some other loss function here.
            pass
    
    def forward(self, preds, labels):
        # loss = []
        # for c in range(preds.shape[-1]):
        #     c_preds = preds[:, :, :, c]
        #     c_labels = labels[:, :, :, c]
        #     loss.append(self._lambda_value * self._main_loss(c_preds, c_labels, self._pos_weight[c]) + \
        #     (1 - self._lambda_value) * self._other_loss(c_preds, c_labels))
        loss = self._lambda_value * self._main_loss(preds, labels, self._pos_weight[0]) + \
            (1 - self._lambda_value) * self._other_loss(preds, labels)
            # AutomaticWeightedLoss
            # loss += self._auto_loss(self._main_loss(c_preds, c_labels, self._pos_weight[c]), self._other_loss(c_preds, c_labels))
        # loss = self._auto_loss(loss)
        return loss

class ClassificationMetrics(nn.Module):
    def __init__(self, threshold:list):
        super(ClassificationMetrics, self).__init__()
        self._threshold = threshold
    
    def _round(self, preds):
        preds = torch.sigmoid(preds) # value in [0, 1]
        for idx, threshold in enumerate(self._threshold):
            preds[:, :, :, idx] = torch.where(preds[:, :, :, idx] >= threshold, 1, 0)

        return preds
    
    def _micro_macro_f1(self, preds, labels):
        # Macro f1
        TP_lst, FN_lst, FP_lst = [], [], []
        # Micro F1
        F1_lst = []
        for c in range(preds.shape[-1]):
            c_preds = preds[:, :, :, c].flatten()
            c_labels = labels[:, :, :, c].flatten()

            TN, FP, FN, TP = confusion_matrix(c_preds, c_labels, labels=[0, 1]).ravel()
            TP_lst.append(TP)
            FN_lst.append(FN)
            FP_lst.append(FP)
            F1_lst.append(2 * TP / (2 * TP + FN + FP))

        macro_f1 = 2 * sum(TP_lst) / (2 * sum(TP_lst) + sum(FN_lst) + sum(FP_lst))
        micro_f1 = sum(F1_lst) / len(F1_lst)

        # ic(classification_report(preds.flatten(), labels.flatten(), labels=np.unique(labels.flatten())))
        # ic(confusion_matrix(preds.flatten(), labels.flatten()))
        return (
            micro_f1,
            macro_f1,
            F1_lst
        )

    def forward(self, preds, labels):
        preds = self._round(preds)

        preds = preds.cpu()
        labels = labels.cpu()

        return self._micro_macro_f1(preds, labels) 

