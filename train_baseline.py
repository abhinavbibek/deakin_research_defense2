import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset import PoisonLabelDataset
from data.utils import (
    gen_poison_idx,
    get_bd_transform,
    get_dataset,
    get_loader,
    get_transform,
)
from model.model import LinearModel
from model.utils import (
    get_network,
    get_optimizer,
    get_scheduler,
)
from utils.setup import (
    get_logger,
    get_saved_dir,
    get_storage_dir,
    load_config,
    set_seed,
)
from utils.trainer.log import result2csv

def train(model, train_loader, criterion, optimizer, epoch, logger, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i, batch in enumerate(train_loader):
        inputs = batch["img"].to(device)
        targets = batch["target"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    acc = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
    return {"loss": avg_loss, "acc": acc}

def test(model, test_loader, criterion, device, logger, prefix="Test"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["img"].to(device)
            targets = batch["target"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    logger.info(f"{prefix} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    return {"loss": avg_loss, "acc": acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/baseline_asd.yaml")
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()

    # Load config
    config, inner_dir, config_name = load_config(args.config)
    args.saved_dir, args.log_dir = get_saved_dir(config, inner_dir, config_name, args.resume)
    args.saved_dir = args.saved_dir.replace("asd", "no_defense") # Separate save dir
    args.log_dir = args.log_dir.replace("asd", "no_defense")
    
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup logger and seed
    set_seed(**config["seed"])
    logger = get_logger(args.log_dir, "no_defense.log", args.resume)
    
    # Load Data
    logger.info("loading data...")
    bd_config = config["backdoor"]
    bd_transform = get_bd_transform(bd_config)
    target_label = bd_config["target_label"]
    poison_ratio = bd_config["poison_ratio"]

    # Transforms
    train_transform = {
        "pre": get_transform(config["transform"]["pre"]),
        "primary": get_transform(config["transform"]["train"]["primary"]),
        "remaining": get_transform(config["transform"]["train"]["remaining"])
    }
    test_transform = {
        "pre": get_transform(config["transform"]["pre"]),
        "primary": get_transform(config["transform"]["test"]["primary"]),
        "remaining": get_transform(config["transform"]["test"]["remaining"])
    }

    # Datasets
    clean_train_data = get_dataset(config["dataset_dir"], train_transform, prefetch=config["prefetch"])
    clean_test_data = get_dataset(config["dataset_dir"], test_transform, train=False, prefetch=config["prefetch"])

    # Create Poisoned Train Set
    poison_train_idx = gen_poison_idx(clean_train_data, target_label, poison_ratio)
    poison_train_data = PoisonLabelDataset(clean_train_data, bd_transform, poison_train_idx, target_label)
    
    # Create Poisoned Test Set
    poison_test_idx = gen_poison_idx(clean_test_data, target_label) # 100% poisoned for ASR calculation
    poison_test_data = PoisonLabelDataset(clean_test_data, bd_transform, poison_test_idx, target_label)

    # Loaders
    train_loader = get_loader(poison_train_data, config["loader"], shuffle=True)
    clean_test_loader = get_loader(clean_test_data, config["loader"])
    poison_test_loader = get_loader(poison_test_data, config["loader"])

    # Model
    backbone = get_network(config["network"])
    model = LinearModel(backbone, backbone.feature_dim, config["num_classes"])
    model = model.to(device)

    # Training Components
    optimizer = get_optimizer(model, config["optimizer"])
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer, config["lr_scheduler"]) # Might be null

    # Training Loop
    best_acc = 0
    logger.info("Start Training No Defense Baseline...")
    
    for epoch in range(config["num_epochs"]):
        train_result = train(model, train_loader, criterion, optimizer, epoch, logger, device)
        clean_result = test(model, clean_test_loader, criterion, device, logger, "Clean Test")
        poison_result = test(model, poison_test_loader, criterion, device, logger, "Poison Test (ASR)")
        
        if scheduler:
            scheduler.step()

        # Save results
        result = {
            "train": train_result,
            "clean_test": clean_result,
            "poison_test": poison_result
        }
        result2csv(result, args.log_dir)

        # Checkpoint
        is_best = clean_result["acc"] > best_acc
        if is_best:
            best_acc = clean_result["acc"]
            torch.save(model.state_dict(), os.path.join(args.saved_dir, "best_model.pt"))
            logger.info(f"New Best Accuracy: {best_acc:.2f}%")
        
        torch.save(model.state_dict(), os.path.join(args.saved_dir, "latest_model.pt"))

    logger.info("Training Complete.")

if __name__ == "__main__":
    main()
