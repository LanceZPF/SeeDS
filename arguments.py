import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='uec256', help='coco, voc')
    parser.add_argument('--dataroot', default='./data/uec256', help='path to dataset')
    # parser.add_argument('--class_embedding', default='VOC/fasttext_synonym.npy')
    # parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
    parser.add_argument('--class_embedding', default='UEC256/clip_uec.npy')
    parser.add_argument('--attSize', type=int, default=512, help='size of semantic features')
    parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')

    parser.add_argument('--syn_num', type=int, default=500, help='50 df. number features to generate per class')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--nz', type=int, default=300, help='size of the latent z vector')

    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
    parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train GAN')
    parser.add_argument('--nepoch_cls', type=int, default=25, help='number of epochs to train CLS')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--cls_weight', type=float, default=0.01, help='weight of the classification loss')
    parser.add_argument('--cls_weight_unseen', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to train GANs ')
    # parser.add_argument('--lr_cls', type=float, default=1e-4, help='learning rate to train CLS ')
    parser.add_argument('--lr_cls', type=float, default=5e-5, help='learning rate to train CLS ')
    parser.add_argument('--testsplit', default='test_0.6_0.3', help='unseen classes feats and labels paths')
    parser.add_argument('--trainsplit', default='train_0.6_0.3', help='seen classes feats and labels paths')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--lz_ratio', type=float, default=0.95, help='mode seeking loss weight')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrain_classifier',
                        default='/media/lance/NewPlant/Users/Lance/Desktop/FD/ZSFD/Desktop/RRFS-main-ori/mmdetection/tools/work_dirs/uecfood256/epoch_12.pth',
                        help="path to pretrain classifier (for seen classes loss on fake features)")
    parser.add_argument('--pretrain_classifier_unseen', default=None,
                        help="path to pretrain classifier (for unseen classes loss on fake features)")
    parser.add_argument('--netG', default=None, help="path to netG (to continue training)")
    # parser.add_argument('--netG', default='/home/lance/Desktop/RRFS-main-new/checkpoints/gen_best.pth')
    parser.add_argument('--netD', default=None, help="path to netD (to continue training)")
    # parser.add_argument('--netD', default='/home/lance/Desktop/RRFS-main-new/checkpoints/disc_best.pth')
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--classes_split', default='')
    parser.add_argument('--outname', default='./checkpoints_try/', help='folder to output data and model checkpoints')
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nclass_all', type=int, default=257, help='number of all classes')
    parser.add_argument('--lr_step', type=int, default=30, help='number of training epoches that lr X gama(e.g.0.1)')
    parser.add_argument('--gan_epoch_budget', type=int, default=38000,
                        help='random pick subset of features to train GAN')

    ##intra contra
    parser.add_argument('--lambda_contra', type=float, default=0.001, help='weight for contrastive loss')
    parser.add_argument('--num_negative', type=int, default=10, help='number of latent negative samples')
    parser.add_argument('--radius', type=float, default=0.0000001, help='positive sample - distance threshold')
    parser.add_argument('--tau', type=float, default=0.1, help='temprature')
    # parser.add_argument('--featnorm', action='store_true', help='whether featnorm')

    ##inter contra
    parser.add_argument('--inter_temp', type=float, default=0.1, help='inter temperature')
    parser.add_argument('--inter_weight', type=float, default=0.001,
                        help='weight of the classification loss when learning G')

    parser.add_argument('--adj_file1', default='/home/lance/Desktop/RRFS-main-new/mmdetection/adj/voc_adj.pkl')
    parser.add_argument('--adj_file2', default='/home/lance/Desktop/RRFS-main-new/mmdetection/adj/voc_hyp_adj.pkl')
    parser.add_argument('--adj_file3', default='/home/lance/Desktop/RRFS-main-new/mmdetection/adj/voc_out_adj.pkl')
    parser.add_argument('--inp_name', default='/home/lance/Desktop/RRFS-main-new/VOC/fasttext_synonym.npy')

    parser.add_argument('--glambda1', type=float, default=0.6)
    parser.add_argument('--glambda2', type=float, default=0.7)

    # parser.add_argument('--embed_dim', default=64, type=int)  # same to z_channels
    parser.add_argument('--n_embed', default=2, type=int)  # num of part segment

    # transformer
    parser.add_argument('--n_layers', default=3, type=int)  # number of layers for transformer
    parser.add_argument('--n_emb_style', default=1, type=int)  # style_protype number
    parser.add_argument('--d_model', default=64, type=int)
    parser.add_argument('--nhead', default=8, type=int)  # number of head
    parser.add_argument('--dim_feedforward', default=512, type=int)  # 12 head -> 768

    # mixer
    parser.add_argument('--z_channels', default=64, type=int)

    parser.add_argument('--din_channels', default=1, type=int)
    parser.add_argument('--dmodel_channels', default=64, type=int)
    parser.add_argument('--dout_channels', default=1, type=int)

    parser.add_argument('--linear_start', default=8.5e-4)
    parser.add_argument('--linear_end', default=1.2e-2)

    parser.add_argument('--is_y_cond', default=True, help='use label condition or not')
    parser.add_argument('--rtdl_params', default={'d_layers': [
        512,
        1024,
        1024,
        1024,
        1024,
        512, ], 'dropout': 0.0}, help='MLP parameters')
    parser.add_argument('--dim_t', default=512, help='MLP parameters')

    parser.add_argument('--syn_time', default=2, help='slices of generation num to avoid runtimeerror')

    opt = parser.parse_args()
    return opt
