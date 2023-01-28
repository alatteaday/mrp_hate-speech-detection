import torch
import random


def prepare_gts(args, max_len, bi_rats_str):
    gts = []
    for bi_rat_str in bi_rats_str:
        bi_list = bi_rat_str.split(',')
        bi_rat = [int(b) for b in bi_list]
        
        if args.intermediate == 'rp':
            bi_rat = [0]+bi_rat
            n_pads = max_len - len(bi_rat)  # num of eos + pads
            bi_gt = bi_rat + [0]*n_pads
        elif args.intermediate == 'mrp':
            bi_gt = [0]+bi_rat+[0]

        gts.append(bi_gt)

    return gts


###### MRP ######
def make_masked_rationale_label(args, labels, emb_layer):
    label_reps_list = []
    masked_idxs_list = []
    masked_labels_list = []
    for label in labels:
        idxs = list(range(len(label)))
        if args.test:
            masked_idxs = idxs[1:-1]
            masked_label = [-100]+label[1:-1]+[-100]
            label_rep = torch.zeros(len(label), emb_layer.embedding_dim)
        else:  # Validation and Training
            masked_idxs = random.sample(idxs[1:-1], int(len(idxs[1:-1])*args.mask_ratio))
            masked_idxs.sort()
            label_tensor = torch.tensor(label).to(args.device)
            label_rep = emb_layer(label_tensor)
            label_rep[0] = torch.zeros(label_rep[0].shape)
            label_rep[-1] = torch.zeros(label_rep[-1].shape)
            for i in masked_idxs:
                label_rep[i] = torch.zeros(label_rep[i].shape)
            
            # For loss
            masked_label = []
            for j in idxs:
                if j in masked_idxs:
                    masked_label.append(label[j])
                else:
                    masked_label.append(-100)
            
        assert len(masked_label) == label_rep.shape[0], '[!] len(masked_label) != label_rep.shape[0] | \n{} \n{}'.format(masked_label, label_rep)
        
        masked_idxs_list.append(masked_idxs)
        masked_labels_list.append(masked_label)
        label_reps_list.append(label_rep)

    return masked_idxs_list, label_reps_list, masked_labels_list
    

def add_pads(args, max_len, labels, masked_labels, label_reps):
    assert len(labels) == len(masked_labels) == len(label_reps), '[!] add_pads | different total nums {} {} {}'.format(len(labels), len(masked_labels), len(label_reps))
    labels_pad, masked_labels_pad, label_reps_pad = [], [], []
    for label, mk_label, rep in zip(labels, masked_labels, label_reps):
        assert len(label) == len(mk_label) == rep.shape[0], '[!] add_pads | different lens of each ele {} {} {}'.format(len(label), len(mk_label), rep.shape[0])
        if args.test:
            labels_pad.append(label)
            masked_labels_pad.append(mk_label)
            label_reps_pad.append(rep)
        else:
            n_pads = max_len - len(label)
            label = label + [0]*n_pads
            mk_label = mk_label + [-100]*n_pads
            zero_ten = torch.zeros(n_pads, 768).to(args.device)
            rep = torch.cat((rep, zero_ten), 0)
            
            assert len(label) == len(mk_label) == rep.shape[0], '[!] add_pads | different lens of each ele'
            labels_pad.append(label)
            masked_labels_pad.append(mk_label)
            label_reps_pad.append(rep)

    return labels_pad, masked_labels_pad, label_reps_pad
