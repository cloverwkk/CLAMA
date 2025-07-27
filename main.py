import os
import csv
import random
import logging
from datetime import datetime
from os.path import exists

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import umap

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from transformers import CLIPModel, CLIPProcessor

import clip
from tqdm import tqdm
import argparse
from sam import SAM


from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)

from data_process_scripts.process_cwru import CWRU_Dataset
from data_process_scripts.process_mfpt import MFPT_Dataset
from data_process_scripts.process_jnu import JNU_Dataset
from data_process_scripts.process_seu import SEU_Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)

def save_results_to_csv(args, eval_results, csv_path="./results2.csv"):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = exists(csv_path)

    rounded_results = {
        k: round(v, 4) if isinstance(v, float) else v 
        for k, v in eval_results.items()
    }
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = [
            'dataname', 'condition', 'lambda', 'rho', 
            'accuracy', 'precision', 'recall', 'f1', 'auc'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            logging.info(f"Created new CSV file: {csv_path}")

        row_data = {
            'dataname': args.data_name,
            'condition': args.con_index,
            'lambda': args.lamb,
            'rho': args.rho,
            'accuracy': rounded_results['accuracy'],
            'precision': rounded_results['precision'],
            'recall': rounded_results['recall'],
            'f1': rounded_results['f1'],
            'auc': rounded_results['auc']
        }

        writer.writerow(row_data)


def adjust_learning_rate(optimizer, scheduler, epoch):
    scheduler.step()
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print(f"Epoch {epoch}: Learning Rate = {current_lr:.8f}")


def check_for_nan(images, text_tokens):
    if torch.isnan(images).any():
        print("Warning: NaN detected in images!")
        print(f"Images tensor: {images}")
    
    if torch.isnan(text_tokens).any():
        print("Warning: NaN detected in text_tokens!")
        print(f"Text tokens tensor: {text_tokens}")


def _convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def image_to_text_contrastive_loss_from_batch(image_features, text_features, labels, temperature=1.0):

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    pos_mask = labels == 0
    neg_mask = labels == 1

    cf_pos = image_features[pos_mask]  
    cf_neg = image_features[neg_mask] 
    at_pos = text_features[pos_mask]   
    at_neg = text_features[neg_mask]  

    def cosine_sim(a, b):
        return F.cosine_similarity(a, b, dim=-1) / temperature

    sim_pos_pos = cosine_sim(cf_pos, at_pos)
    sim_pos_neg = cosine_sim(cf_pos, at_neg)
    log_prob_pos = -torch.log(torch.exp(sim_pos_pos) / (torch.exp(sim_pos_pos) + torch.exp(sim_pos_neg)))

    sim_neg_neg = cosine_sim(cf_neg, at_neg)
    sim_neg_pos = cosine_sim(cf_neg, at_pos)
    log_prob_neg = -torch.log(torch.exp(sim_neg_neg) / (torch.exp(sim_neg_neg) + torch.exp(sim_neg_pos)))

    loss = (log_prob_pos.mean() + log_prob_neg.mean()) / 2
    return loss


class CLIPDataProcessor:
    def __init__(self, preprocess, image_size=224, n_fft=64, hop_length=16):
        self.normal_text = "The machine is in normal state"
        self.fault_text = "The machine is in fault state"
        self.image_size = image_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.processor = preprocess

    def _signal_to_spectrogram(self, signal_batch):
        """Convert 1D signal batch to spectrogram images"""
        # signal_batch shape: (batch_size, 1, 1024)
        batch_size, _, _ = signal_batch.shape
        signal_batch = signal_batch.squeeze(1)  # (batch_size, 1024)
        
        # Compute STFT
        window = torch.hann_window(self.n_fft).to(signal_batch.device)
        spec = torch.stft(signal_batch, 
                         n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         window=window,
                         return_complex=True)
        
        # Get magnitude and normalize
        magnitude = torch.abs(spec)
        min_val = magnitude.amin(dim=(1,2), keepdim=True)
        max_val = magnitude.amax(dim=(1,2), keepdim=True)
        magnitude = (magnitude - min_val) / (max_val - min_val + 1e-6)
        
        # Convert to 3-channel "image"
        images = magnitude.unsqueeze(1).repeat(1,3,1,1)  # (batch_size, 3, H, W)

        return images

    def _signal_to_waveform_image(self, signal_batch):
        """Convert 1D signal batch to waveform images matching the new plot style"""
        batch_size, _, signal_length = signal_batch.shape
        signal_batch = signal_batch.squeeze(1).cpu().numpy()  # (batch_size, signal_length)
        
        images = []
        for i in range(batch_size):
            signal = signal_batch[i]

            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            ax.plot(signal, color='blue', linewidth=1)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            plt.tight_layout()

            fig.canvas.draw()
            image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            plt.close(fig)

            image = Image.fromarray(image)

            if i == 0:
                image.save("./waveform_sample.png")

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            image = transform(image)  # (3, image_size, image_size)
            
            images.append(image)
        
        return torch.stack(images)  # (batch_size, 3, image_size, image_size)

    def _apply_augmentation(self, signal, method):
        """Apply specified augmentation to 1D signal"""
        if method == 'amplify':
            return signal * random.uniform(1.5, 2.0)
        elif method == 'shrink':
            return signal * random.uniform(0.2, 0.5)
        elif method == 'random_region':
            signal = signal.clone()
            length = signal.shape[-1]
            region_size = length // 20

            start1 = random.randint(0, length - 2 * region_size)

            signal[..., start1:start1 + region_size] *= random.uniform(3.0, 5.0)

            return signal

    def process_batch(self, data_batch, device):
        """
        Process batch of signals:
        Returns:
            images: Tensor of shape (2*N, 3, H, W)
            text_inputs: Dictionary with tokenized text
        """
        # Generate normal samples
        normal_images = self._signal_to_waveform_image(data_batch)
        
        # Generate anomaly samples with random augmentation
        aug_methods = ['amplify', 'shrink', 'random_region']
        augmented_signals = torch.stack([self._apply_augmentation(x, random.choice(aug_methods)) for x in data_batch])
        anomaly_images = self._signal_to_waveform_image(augmented_signals)
        
        # Combine images and texts
        all_images = torch.cat([normal_images, anomaly_images]).to(device)
        all_texts = [self.normal_text]*len(data_batch) + [self.fault_text]*len(data_batch)
        
        # Tokenize texts
        text_inputs = clip.tokenize(all_texts).to(device)
        
        return all_images, text_inputs

    def process_test_batch(self, data_batch, labels):
        """Process test batch with actual labels"""
        images = self._signal_to_waveform_image(data_batch)
        texts = [self.normal_text if l == 0 else self.fault_text for l in labels]
        text_inputs = self.processor(text=texts, 
                                   return_tensors="pt", 
                                   padding=True,
                                   truncation=True).to(data_batch.device)
        return images, text_inputs

def train(args, model, processor, train_loader, test_loader, optimizer, device):
    best_acc = 0.0
    model.train()
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer.base_optimizer, step_size=20, gamma=0.5)

    for epoch in range(args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (data, target, _) in enumerate(progress_bar):
            data = data.to(device).float()

            if data.shape[1:] == (1024, 1):
                data = data.permute(0, 2, 1)  # (batch_size, 1, 1024)

            # Prepare input
            images, text_tokens = processor.process_batch(data, device)
            batch_size_half = data.size(0)
            ground_truth = torch.arange(2*batch_size_half, dtype=torch.long, device=device)
            labels = torch.cat([torch.zeros(batch_size_half), torch.ones(batch_size_half)]).to(device)

            # First forward-backward pass
            logits_per_image, logits_per_text = model(images, text_tokens)
            logits_per_image *= (np.exp(0.01) / np.exp(0.07))
            logits_per_text *= (np.exp(0.01) / np.exp(0.07))

            image_loss = loss_img(logits_per_image, ground_truth)
            text_loss = loss_txt(logits_per_text, ground_truth)
            clip_loss = 0.5 * image_loss + 0.5 * text_loss

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
            contra_loss = image_to_text_contrastive_loss_from_batch(image_features, text_features, labels, temperature=1)

            loss = clip_loss + args.lamb * contra_loss

            optimizer.zero_grad()
            loss.backward()
            _convert_models_to_fp32(model)
            optimizer.first_step(zero_grad=True)

            # Second forward-backward pass
            # Forward again after SAM step
            logits_per_image, logits_per_text = model(images, text_tokens)
            logits_per_image *= (np.exp(0.01) / np.exp(0.07))
            logits_per_text *= (np.exp(0.01) / np.exp(0.07))

            image_loss = loss_img(logits_per_image, ground_truth)
            text_loss = loss_txt(logits_per_text, ground_truth)
            clip_loss = 0.5 * image_loss + 0.5 * text_loss

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)
            contra_loss = image_to_text_contrastive_loss_from_batch(image_features, text_features, labels, temperature=1)

            loss = clip_loss + args.lamb * contra_loss

            loss.backward()
            optimizer.second_step(zero_grad=True)
            clip.model.convert_weights(model)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        adjust_learning_rate(optimizer.base_optimizer, scheduler, epoch)

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        logging.info(f"Train Loss: {avg_loss:.4f}")

    logging.info(f"Saved model to {args.model_dir}/clip.pth")
    print(f"Saved model")
    

def evaluate(model, processor, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    normal_text = processor.normal_text
    fault_text = processor.fault_text

    text_inputs = clip.tokenize([normal_text, fault_text]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        for data, labels, _ in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device).float()
            
            if data.dim() == 3 and data.shape[1] == 1 and data.shape[2] == 1024:
                data = data.permute(0, 2, 1)  # (batch_size, 1, 1024)
            
            images = processor._signal_to_waveform_image(data).to(device)
            
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()

            preds = logits_per_image.argmax(dim=1).cpu().numpy()
            scores = torch.softmax(logits_per_image, dim=1)[:, 1].cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_scores.extend(scores)

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    logging.info("\nEvaluation Results:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info("Confusion Matrix:")
    logging.info(conf_matrix)
    logging.info(f"AUC: {roc_auc:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix,
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr
    }


def visualize_tsne(model, processor, test_loader, device, save_path=None):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for data, labels, _ in tqdm(test_loader, desc="Extracting Features for t-SNE"):
            data = data.to(device).float()

            if data.dim() == 3 and data.shape[1] == 1 and data.shape[2] == 1024:
                data = data.permute(0, 2, 1)

            images = processor._signal_to_waveform_image(data).to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_features.append(image_features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    plt.figure(figsize=(8, 6))
    for label in np.unique(all_labels):
        indices = all_labels == label
        label_name = 'Normal' if label == 0 else 'Fault'
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label_name, alpha=0.6)

    plt.legend()
    plt.title("t-SNE Visualization of Image Features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="CLIP Training on CWRU Dataset")
    parser.add_argument("--data_name", type=str, default="MFPT")
    parser.add_argument("--model_dir", type=str, default="./models")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lamb", type=float, default=0.003)
    parser.add_argument("--vision_encoder", type=str, default="ViT-L/14")
    parser.add_argument("--normlizetype", type=str, default="-1-1")
    parser.add_argument("--con_index", type=int, default=2)
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--rho", default=0.0001, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use (default: 0)")

    args = parser.parse_args()

    os.makedirs("./logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logs/training_{args.data_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info("Starting training with parameters:")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    # Device setup
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    # Initialize CLIP model
    print(f"Loading CLIP with vision encoder {args.vision_encoder}")
    model, preprocess = clip.load(args.vision_encoder, device=device, jit=False)
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    
    # Initialize dataset and data processor
    if args.data_name == "CWRU":
        train_dataset, val_dataset = CWRU_Dataset("data/cwru", args.normlizetype, args.con_index).data_prepare()
    elif args.data_name == "MFPT":
        train_dataset, val_dataset = MFPT_Dataset("data/mfpt", args.normlizetype).data_prepare()
    elif args.data_name == "JNU":
        train_dataset, val_dataset = JNU_Dataset("data/jnu", args.normlizetype, args.con_index).data_prepare()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    processor = CLIPDataProcessor(preprocess=preprocess)

    # Optimizer setup
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_sgd = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    base_optimizer = torch.optim.SGD
    optimizer_sam = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)

    try:
        logger.info("Starting training...")
        train(args, model, processor, train_loader, test_loader, optimizer_sam, device)
        
        logger.info("Starting evaluation...")
        eval_results = evaluate(model, processor, test_loader, device)
        save_results_to_csv(args, eval_results)

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()