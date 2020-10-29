# Loss实现

# 参考 
http://aiuai.cn/aifarm1330.html
DiceLoss https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py 



# custom loss function example
```python

class MyLoss(nn.Module):
    """
    l2正则
    """

    def __init__(self):
        super(MyLoss, self).__init__()
        print('Using customize loss')

    def forward(self, pred, truth):
        index_malignant = (truth == 1).nonzero().squeeze()
        index_benign = (truth == 0).nonzero().squeeze()

        pred_benign = pred.index_select(0, index_benign)
        pred_malignant = pred.index_select(0, index_malignant)

        l2r_b = torch.exp(F.cross_entropy(pred_benign, truth.index_select(0, index_benign)))
        l2r_m = torch.exp(F.cross_entropy(pred_malignant, truth.index_select(0, index_malignant)))
        print(l2r_b, l2r_m)
        lambda_loss = 0.5
        loss = lambda_loss / 2 * (l2r_b * l2r_b + l2r_m * l2r_m)
        return loss
```