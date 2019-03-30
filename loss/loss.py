import torch
import torch.nn.functional as F

def categorical_cross_entropy(pred, target, weight=None, size_average=None):
    # Make sure we're dealing with the same size here
    n, c, h, w = pred.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
    pred = pred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(pred, target.long(), weight=weight, size_average=size_average, ignore_index=250)
    return loss

#     if pred.shape != target.shape:
#         pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=True).squeeze()
#     pred = torch.tensor(np.argmax(np.transpose(pred.detach().numpy(), (1, 2, 0)), axis=2), dtype=torch.float).unsqueeze(0)

#     return F.cross_entropy(pred, target, weight=weight, size_average=size_average)

def multi_scale_categorical_cross_entropy(pred, target, scale_weights=[1.0, 0.4, 0.16], weight=None, size_average=None):
    loss = 0
    for p, t, scale_w in zip(pred, target, scale_weights):
        loss = loss + scale_w * categorical_cross_entropy(p, t, weight=weight, size_average=size_average)
    return loss
    