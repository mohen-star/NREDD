import argparse
import numpy as np
import torch
from check_utils import deal_data_for_remove, resample_swc_, saveswc, loadswc, swc_progress, get_swc_from_edges
from model.check_model import TLP, TED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_point(swc_path, is_start=False):
    swc_block, swc_name, data = deal_data_for_remove(swc_path, is_start=is_start)
    model = TED(3, device=device).to(device)
    ckpt_path = r'check_point\TED.pkl'
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    result = np.zeros((len(data)))
    with torch.no_grad():
        for i in range(len(data)):
            adj, feat_xyz, feat_sp = data[i]
            feat_xyz = feat_xyz.to(device)
            adj = adj.to(device)
            feat_sp = feat_sp.to(device)
            if is_start and i == 0:
                result[i] = 0
                continue
            if len(feat_xyz) <= 2:
                result[i] = 1
                continue

            output = model(feat_xyz, adj, feat_sp)
            output = output.squeeze(dim=0).cpu().detach().numpy()
            result[i] = np.argmax(output, axis=0)
    bound_idx = []
    det_idx = np.where(result == 1)[0]
    if len(det_idx) > 0:
        for det_id in det_idx:
            if np.any(swc_block[det_id, 2:5] <= 13) or np.any(swc_block[det_id, 2:5] >= 244) or np.any(
                    swc_block[det_id, 5] >= 2.5):
                bound_idx.append(det_id)
        det_idx = np.setdiff1d(det_idx, bound_idx)
        print("det_node_idx: ", det_idx)
        swc_block = np.delete(swc_block, det_idx, axis=0)
    swc_block = resample_swc_(swc_block)
    save_remove_path = swc_path.replace('.swc', '._remove.swc')
    saveswc(save_remove_path, swc_block)
    return swc_block


def link_predict(swc, save_path):
    model = TLP().to(device)
    ckpt_path = r'check_point\TLP.pth'
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        feat_xyz, feat_sp, adj, adj_org = swc_progress(swc.copy(), save_path)
        feat_xyz = feat_xyz.to(device)
        feat_sp = feat_sp.to(device)

        adj = adj.to(device)
        adj_org = adj_org.to(device)

        adj = adj.squeeze(dim=0)
        output = model(feat_xyz, adj, feat_sp)
        get_swc_from_edges(output, adj_org, swc, save_path)


if __name__ == '__main__':
    print("Start error_repair...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', '-i', type=str,
                        default=r'..\data\test_data\x18044_y13021_z3647.swc')
    args = parser.parse_args()

    swc_block = loadswc(args.input_path)
    if swc_block.shape[0] >= 15:
        swc_block = remove_point(args.input_path, is_start=False)
        save_path = args.input_path
        link_predict(swc_block, save_path)
        print("Repair Finished")
        print("The result has been saved at {}".format(save_path.replace('.swc', '._repair.swc')))
