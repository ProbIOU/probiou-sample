import torch

def gbb_form(boxes):
    return torch.cat((boxes[:,:2],torch.pow(boxes[:,2:4],2)/12,boxes[:,4:]),1)

def rotated_form(a_, b_, angles):

    a   = a_*torch.pow(torch.cos(angles),2.)+b_*torch.pow(torch.sin(angles),2.)
    b   = a_*torch.pow(torch.sin(angles),2.)+b_*torch.pow(torch.cos(angles),2.)
    c   = a_*torch.cos(angles)*torch.sin(angles)-b_*torch.sin(angles)*torch.cos(angles)
    return a,b,c

def probiou_loss(pred, target, eps = 1e-3, l='l1'):
    """
        pred    -> a matrix [N,5](x,y,w,h,angle) containing ours predicted box ;in case of HBB angle == 0
        target  -> a matrix [N,5](x,y,w,h,angle) containing ours target    box ;in case of HBB angle == 0
        eps     -> threshold to avoid infinite values
        l       -> ('l1' in [0,1] or 'l2' in [0,inf]) metrics according our paper

    """


    gbboxes1 = gbb_form(pred)
    gbboxes2 = gbb_form(target)

    x1, y1, a1_, b1_, c1_ = gbboxes1[:,0], gbboxes1[:,1], gbboxes1[:,2], gbboxes1[:,3], gbboxes1[:,4]
    x2, y2, a2_, b2_, c2_ = gbboxes2[:,0], gbboxes2[:,1], gbboxes2[:,2], gbboxes2[:,3], gbboxes2[:,4]

    a1, b1, c1 = rotated_form(a1_, b1_, c1_)
    a2, b2, c2 = rotated_form(a2_, b2_, c2_)

    t1  = (((a1+a2)*(torch.pow(y1-y2,2)) + (b1+b2)*(torch.pow(x1-x2,2)) )/((a1+a2)*(b1+b2)-(torch.pow(c1+c2,2))+eps))*0.25
    t2  = (((c1+c2)*(x2-x1)*(y1-y2))/((a1+a2)*(b1+b2)-(torch.pow(c1+c2,2))+eps))*0.5
    t3  = torch.log(((a1+a2)*(b1+b2)-(torch.pow(c1+c2,2)))/(4*torch.sqrt((a1*b1-torch.pow(c1,2))*(a2*b2-torch.pow(c2,2)))+eps)+eps)*0.5

    B_d = t1 + t2 + t3

    B_d = torch.clamp(B_d,eps,100.0)
    l1  = torch.sqrt(1.0-torch.exp(-B_d)+eps)
    l_i  = torch.pow(l1, 2.0)
    l2  = -torch.log(1.0 - l_i+eps)

    if l=='l1':
        probiou = l1
    if l=='l2':
        probiou = l2

    return probiou

def main():

    P   = torch.rand(8,5)
    T   = torch.rand(8,5)
    LOSS        = probiou_loss(P, T)
    REDUCE_LOSS = torch.mean(LOSS)
    print(REDUCE_LOSS.item())

if __name__ == '__main__':
    main()
