import csv
import glob
import json
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import yaml
import torch

from networks import define_G


def normalize_obs(obs):
    # default normalize
    cav_lb = [400, 42, 20, 80]
    cav_ub = [900, 50, 40, 100]
    relative_bv_lb = [-30, -8, 20, 80]
    relative_bv_ub = [30, 8, 40, 100]
    total_lb = cav_lb + relative_bv_lb * 6
    total_ub = cav_ub + relative_bv_ub * 6
    total_lb = np.array(total_lb)
    total_ub = np.array(total_ub)
    normalized_obs = np.copy(obs)

    # center at CAV
    noneveh_index = np.where(normalized_obs == -1)
    normalized_obs[:, 0::4][:, 1:] = normalized_obs[:, 0::4][:, 1:] - np.repeat(normalized_obs[:, 0].reshape(-1, 1), 6, axis=1)
    normalized_obs[:, 1::4][:, 1:] = normalized_obs[:, 1::4][:, 1:] - np.repeat(normalized_obs[:, 1].reshape(-1, 1), 6, axis=1)
    normalized_obs[noneveh_index] = -1

    # out of bound vehicles are [-1]*4, others follow default normalize
    _1_bv_outbound = np.where(abs(normalized_obs[:,4])>relative_bv_ub[0])
    _2_bv_outbound = np.where(abs(normalized_obs[:,8])>relative_bv_ub[0])
    _3_bv_outbound = np.where(abs(normalized_obs[:,12])>relative_bv_ub[0])
    _4_bv_outbound = np.where(abs(normalized_obs[:,16])>relative_bv_ub[0])
    _5_bv_outbound = np.where(abs(normalized_obs[:,20])>relative_bv_ub[0])
    _6_bv_outbound = np.where(abs(normalized_obs[:,24])>relative_bv_ub[0])

    # normalize the data from -1 to 1
    normalized_obs = (normalized_obs - total_lb) / (total_ub - total_lb) * 2 - 1

    # constrain the data to be in the range of -1 to 1
    normalized_obs = np.clip(normalized_obs, -1, 1)
    normalized_obs[_1_bv_outbound,4:8] = np.array([-1]*4).reshape(1,-1)
    normalized_obs[_2_bv_outbound,8:12] = np.array([-1]*4).reshape(1,-1)
    normalized_obs[_3_bv_outbound,12:16] = np.array([-1]*4).reshape(1,-1)
    normalized_obs[_4_bv_outbound,16:20] = np.array([-1]*4).reshape(1,-1)
    normalized_obs[_5_bv_outbound,20:24] = np.array([-1]*4).reshape(1,-1)
    normalized_obs[_6_bv_outbound,24:28] = np.array([-1]*4).reshape(1,-1)

    normalized_obs[noneveh_index] = -1
    
    # remove cav x
    normalized_obs = normalized_obs[:,1:]

    return normalized_obs


def convert2neuralsmardata(fcd_traj):
    data = []
    total_time_step = len(fcd_traj)

    for step in fcd_traj:
        step_data = {"time": step["@time"], "observation": [-1]*28, "output": 0, "criticality": 0}
        cav_obs = []
        if isinstance(step["vehicle"], dict):
            step["vehicle"] = [step["vehicle"]]
        for v in step["vehicle"]:
            if v["@id"] == "CAV":
                cav_obs = [v["@x"], v["@y"], v["@speed"], v["@angle"]]
                cav_obs = [float(ele) for ele in cav_obs]
                step_data["observation"][0:4] = cav_obs
                break
        assert(cav_obs!=[])
        allbv_obs = []
        for v in step["vehicle"]:
            if v["@id"] != "CAV":
                bv_obs = [v["@x"], v["@y"], v["@speed"], v["@angle"]]
                bv_obs = [float(ele) for ele in bv_obs]
                bv_obs.append((bv_obs[0]-cav_obs[0])**2+(bv_obs[1]-cav_obs[1])**2)
                allbv_obs.append(bv_obs)
        
        sorted_allbv_obs = sorted(allbv_obs,key=lambda x: (x[4]))
        for i in range(6):
            if i < len(sorted_allbv_obs):
                step_data["observation"][4*(i+1):4*(i+2)] = sorted_allbv_obs[i][0:4]
        
        data.append(step_data)
    return data


def convert2list(dict_data):
    data = []
    for step in dict_data:
        data.append(step["observation"])
    return data


def get_test_ep_list(data_folder, crash_flag=True):
    test_ep_fcd_list = []
    test_ep_fcd_info_list = []
    if crash_flag:
        subfolder_name = "crash"
    else:
        subfolder_name = "safe"
    f = os.path.join(data_folder, subfolder_name, "test_ep.fcd.json")
    with open(f, "r") as fp:
        for line in fp:
            info = json.loads(line)
            test_ep_fcd_list.append(int(info["original_name"]))
            test_ep_fcd_info_list.append(info) 
    print(len(test_ep_fcd_list))
    return test_ep_fcd_list, test_ep_fcd_info_list


def prepare_NNMetric_data(yaml_conf_path, checkpoint, data_folder):
    # Load checkpoint
    with open(yaml_conf_path, 'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
    net_G = define_G(yaml_conf = yaml_conf, input_dim=27, output_dim=1, device="cpu")
    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint, map_location="cpu")
        # print(f"Best checkpoint epoch id {checkpoint['best_epoch_id']} with accuracy {checkpoint['best_val_acc']}")
        net_G.load_state_dict(checkpoint['model_G_state_dict'])
        net_G.eval()
    else:
        raise NotImplementedError(
            'pre-trained weights %s does not exist...' % os.path.join(yaml_conf["ckpt"], 'best_ckpt.pt'))

    test_crash_data = np.array([])
    test_safe_data = np.array([])

    raw_test_crash_ep_list, raw_test_crash_ep_fcd_info_list = get_test_ep_list(data_folder, crash_flag=True)
    for fcd_info in tqdm(raw_test_crash_ep_fcd_info_list):
        traj = fcd_info["fcd-export"]["timestep"]
        data = convert2neuralsmardata(traj)
        obs = convert2list(data)
        normalized_obs = np.array(normalize_obs(obs))
        infer_results = baseline_inference(normalized_obs, net_G)
        test_crash_data = np.append(test_crash_data, infer_results)

    raw_test_safe_ep_list, raw_test_safe_ep_fcd_info_list = get_test_ep_list(data_folder, crash_flag=False)
    for fcd_info in tqdm(raw_test_safe_ep_fcd_info_list):
        traj = fcd_info["fcd-export"]["timestep"]
        data = convert2neuralsmardata(traj)
        obs = convert2list(data)
        normalized_obs = np.array(normalize_obs(obs))
        infer_results = baseline_inference(normalized_obs, net_G)
        test_safe_data = np.append(test_safe_data, infer_results)

    print(test_crash_data.shape, test_safe_data.shape)

    output_folder = "dataset/neuralmetric/testing_results_nnmetric"
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "test_crash_data_NNMetric.npy"), np.array(test_crash_data))
    np.save(os.path.join(output_folder, "test_safe_data_NNMetric.npy"), np.array(test_safe_data))


def baseline_inference(x, net_G):
    x = torch.tensor(x, dtype=torch.float32)
    # Apply the inference
    split_x = torch.split(x, 1000)
    final_y_pred = None
    for sec in split_x:
        y_pred = net_G(sec)
        if final_y_pred is None:
            final_y_pred = y_pred.detach().cpu().squeeze().numpy()
        else:
            final_y_pred = np.append(final_y_pred, y_pred.detach().cpu().squeeze().numpy())
    return final_y_pred


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_conf', type=str, metavar='N', help='the yaml configuration file path')
    parser.add_argument('--checkpoint', type=str, metavar='N', help='the checkpoint file path')
    parser.add_argument('--data_folder', type=str, metavar='N', help='the data folder path')
    args = parser.parse_args()

    prepare_NNMetric_data(args.yaml_conf, args.checkpoint, args.data_folder)