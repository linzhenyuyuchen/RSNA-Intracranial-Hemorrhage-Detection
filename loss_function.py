def run_train_valid(foldn):
  #此处省略一万字
  pos_weight = torch.Tensor([2., 1., 1., 1., 1., 1.]).cuda()
  criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
