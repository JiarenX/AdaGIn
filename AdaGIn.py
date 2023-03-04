from parse_args import *
from utils import *
from dgi import *
import torch.nn.functional as F
import loss_func


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    adj_s, feature_s, label_s, idx_tot_s = load_data(dataset=args.source_dataset+'.mat', device=device,
                                                     seed=args.seed, is_blog=args.is_blog)
    adj_t, feature_t, label_t, idx_tot_t = load_data(dataset=args.target_dataset+'.mat', device=device,
                                                     seed=args.seed, is_blog=args.is_blog)
    n_samples = args.n_samples.split(',')
    output_dims = args.output_dims.split(',')
    emb_model = GraphSAGE(**{
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "input_dim": feature_s.shape[1],
        "layer_specs": [
            {
                "n_sample": int(n_samples[0]),
                "output_dim": int(output_dims[0]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[1]),
                "output_dim": int(output_dims[1]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[-1]),
                "output_dim": int(output_dims[-1]),
                "activation": F.relu,
            }
        ],
        "device": device
    }).to(device)
    cly_model = Cly_net(2*int(output_dims[-1]), label_s.shape[1], args.arch_cly).to(device)
    disc_model = Disc(2 * int(output_dims[-1]) * label_s.shape[1], args.arch_disc, 1).to(device)
    # define the optimizers
    total_params = list(emb_model.parameters()) + list(cly_model.parameters()) + list(disc_model.parameters())
    dgi_model = DGI(2 * int(output_dims[-1])).to(device)
    total_params += list(dgi_model.parameters())
    cly_optim = torch.optim.Adam(total_params, lr=args.lr_cly, weight_decay=args.weight_decay)
    lr_lambda = lambda epoch: (1 + 10*float(epoch) / args.epochs)**(-0.75)
    scheduler = torch.optim.lr_scheduler.LambdaLR(cly_optim, lr_lambda=lr_lambda)
    best_micro_f1, best_macro_f1, best_epoch = 0, 0, 0
    num_batch = round(max(feature_s.shape[0]/(args.batch_size/2), feature_t.shape[0]/(args.batch_size/2)))
    for epoch in range(args.epochs):
        s_batches = batch_generator(idx_tot_s, int(args.batch_size/2))
        t_batches = batch_generator(idx_tot_t, int(args.batch_size/2))
        emb_model.train()
        cly_model.train()
        disc_model.train()
        dgi_model.train()
        p = float(epoch) / args.epochs
        grl_lambda = min(2. / (1. + np.exp(-10. * p)) - 1, 0.2)
        for iter in range(num_batch):
            b_nodes_s = next(s_batches)
            b_nodes_t = next(t_batches)
            source_features, cly_loss_s = do_iter(emb_model, cly_model, adj_s, feature_s, label_s, idx=b_nodes_s,
                                                  is_social_net=args.is_social_net)
            target_features, _ = do_iter(emb_model, cly_model, adj_t, feature_t, label_t, idx=b_nodes_t,
                                         is_social_net=args.is_social_net)         
            shuf_idx_s = np.arange(label_s.shape[0])
            np.random.shuffle(shuf_idx_s)
            shuf_feat_s = feature_s[shuf_idx_s, :]
            shuf_idx_t = np.arange(label_t.shape[0])
            np.random.shuffle(shuf_idx_t)
            shuf_feat_t = feature_t[shuf_idx_t, :]
            neg_source_feats = emb_model(b_nodes_s, adj_s, shuf_feat_s)
            logits_s = dgi_model(neg_source_feats, source_features)
            neg_target_feats = emb_model(b_nodes_t, adj_t, shuf_feat_t)
            logits_t = dgi_model(neg_target_feats, target_features)
            labels_dgi = torch.cat([torch.zeros(int(args.batch_size/2)), torch.ones(int(args.batch_size/2))]).unsqueeze(0).to(device)
            dgi_loss = args.dgi_param * (F.binary_cross_entropy_with_logits(logits_s, labels_dgi) + F.binary_cross_entropy_with_logits(logits_t, labels_dgi))
            features = torch.cat((source_features, target_features), 0)
            outputs = cly_model(features)
            softmax_output = nn.Softmax(dim=1)(outputs)
            domain_loss = args.cdan_param * loss_func.CDAN([features, softmax_output], disc_model, None, grl_lambda,
                                                           None, device=device)
            loss = cly_loss_s + dgi_loss + domain_loss
            cly_optim.zero_grad()
            loss.backward()
            cly_optim.step()

        emb_model.eval()
        cly_model.eval()
        cly_loss_bat_s, micro_f1_s, macro_f1_s, embs_whole_s, targets_whole_s = evaluate(emb_model, cly_model, adj_s, feature_s, label_s,
                                                                                         idx_tot_s, args.batch_size, mode='test', is_social_net=args.is_social_net)
        print("epoch {:03d} | source loss {:.4f} | source micro-F1 {:.4f} | source macro-F1 {:.4f}".
              format(epoch, cly_loss_bat_s, micro_f1_s, macro_f1_s))
        cly_loss_bat_t, micro_f1_t, macro_f1_t, embs_whole_t, targets_whole_t = evaluate(emb_model, cly_model, adj_t, feature_t, label_t,
                                                                                         idx_tot_t, args.batch_size, mode='test', is_social_net=args.is_social_net)
        print("target loss {:.4f} | target micro-F1 {:.4f} | target macro-F1 {:.4f}".format(cly_loss_bat_t, micro_f1_t, macro_f1_t))
        if (micro_f1_t + macro_f1_t) > (best_micro_f1 + best_macro_f1):
            best_micro_f1 = micro_f1_t
            best_macro_f1 = macro_f1_t
            best_epoch = epoch
            print('saving model...')
        scheduler.step()
    print("test metrics on target graph:")
    print('---------- random seed: {:03d} ----------'.format(args.seed))
    print("micro-F1 {:.4f} | macro-F1 {:.4f}".format(best_micro_f1, best_macro_f1))


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:' + str(args.device))
    main(args)
