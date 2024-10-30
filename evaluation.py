import logging
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import torch.utils.data

from models.ClientUpdare import validate


@torch.no_grad()
def evaluate(nodes, num_nodes, nets, device, split="test", stacks=2):
    results = defaultdict(lambda: defaultdict(list))
    for node_id in range(num_nodes):  # iterating over nodes
        if split == "test":
            curr_data = nodes.test_loaders[node_id]
        elif split == "val":
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        net = nets[node_id]
        train_loss, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = validate(
            net, curr_data, device, stacks
        )

        results[node_id]["train_loss"] = train_loss / len(curr_data.dataset)
        results[node_id]["smape_loss"] = smape_loss
        results[node_id]["mae_loss"] = mae_loss
        results[node_id]["mse_loss"] = mse_loss
        results[node_id]["rmse_loss"] = rmse_loss
        results[node_id]["r2_loss"] = r2_loss
    return results


def eval_model(nodes, num_nodes, nets, device, split, stacks=2):
    curr_results = evaluate(nodes, num_nodes, nets, device, split=split, stacks=stacks)
    avg_train_loss = np.mean([val["train_loss"] for val in curr_results.values()])
    avg_smape_loss = np.mean([val["smape_loss"] for val in curr_results.values()])
    avg_mae_loss = np.mean([val["mae_loss"] for val in curr_results.values()])
    avg_mse_loss = np.mean([val["mse_loss"] for val in curr_results.values()])
    avg_rmse_loss = np.mean([val["rmse_loss"] for val in curr_results.values()])
    avg_r2_loss = np.mean([val["r2_loss"] for val in curr_results.values()])

    return (
        curr_results,
        avg_train_loss,
        avg_smape_loss,
        avg_mae_loss,
        avg_mse_loss,
        avg_rmse_loss,
        avg_r2_loss,
    )


def save_to_csv(round, data_split, results, save_path):
    results_df = pd.DataFrame(
        columns=["round", "eval_data", "cid", "smape", "mae", "mse", "rmse", "r2"]
    )
    for cid, metrics in results.items():
        row = pd.DataFrame(
            {
                "round": [round],
                "eval_data": [data_split],
                "cid": [cid],
                "smape": [metrics["smape_loss"]],
                "mae": [metrics["mae_loss"]],
                "mse": [metrics["mse_loss"]],
                "rmse": [metrics["rmse_loss"]],
                "r2": [metrics["r2_loss"]],
            }
        )
        results_df = pd.concat([results_df, row], ignore_index=True)
    results_df.sort_values(by=["round", "eval_data", "cid"], inplace=True)
    if os.path.sep in save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False, mode="a", header=round == 0 and data_split == "val")


def eval_and_save(
    nodes,
    num_nodes,
    net_locals,
    device,
    num_stacks,
    step,
    save_path="results.csv",
):
    # Validation data
    (
        step_results,
        avg_train_loss,
        avg_smape_loss,
        avg_mae_loss,
        avg_mse_loss,
        avg_rmse_loss,
        avg_r2_loss,
    ) = eval_model(
        nodes,
        num_nodes,
        net_locals,
        device,
        split="val",
        stacks=num_stacks,
    )
    save_to_csv(
        round=step, data_split="val", results=step_results, save_path=save_path
    )
    # Test data
    (
        step_results,
        avg_train_loss,
        avg_smape_loss,
        avg_mae_loss,
        avg_mse_loss,
        avg_rmse_loss,
        avg_r2_loss,
    ) = eval_model(
        nodes,
        num_nodes,
        net_locals,
        device,
        split="test",
        stacks=num_stacks,
    )
    logging.info(
        f"\nStep: {step + 1}, AVG Loss: {avg_train_loss:.4f},  AVG SMAPE: {avg_smape_loss:.4f}, AVG MAE: {avg_mae_loss:.4f}, AVG MSE: {avg_mse_loss:.4f}, AVG RMSE: {avg_rmse_loss:.4f}, AVG R2: {avg_r2_loss:.4f}"
    )
    save_to_csv(
        round=step, data_split="test", results=step_results, save_path=save_path
    )
