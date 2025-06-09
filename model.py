import torch
import torch.nn.functional as F
import torch.nn.functional as f
from torch import nn
from transformers import ViTConfig, ViTFeatureExtractor, ViTForImageClassification, ViTModel

from config import get_config


class SnowRanker(nn.Module):
    def __init__(self):
        super().__init__()
        config = get_config()
        self.name = "Self-made CNN"

        image_height = config["IMAGE"]["HEIGHT"]
        image_width = config["IMAGE"]["WIDTH"]
        kernel_size = config["MODEL"]["SNOWRANKER"]["KERNEL_SIZE"]
        num_channels = 6 if config["IMAGE"]["REFERENCE_IMAGE"] else 3

        self.layers = nn.ModuleList()

        # Add cnn layers
        for _ in range(3):
            self.layers.append(nn.Conv2d(num_channels, num_channels * 3, kernel_size, 1, kernel_size // 2))
            self.layers.append(nn.MaxPool2d(2, 2))
            self.layers.append(nn.BatchNorm2d(num_channels * 3))
            self.layers.append(nn.GELU())
            num_channels *= 3
            image_height, image_width = image_height // 2, image_width // 2

        self.layers.append(nn.Flatten())
        # Add fc layers
        num_neurons = image_height * image_width * num_channels
        self.layers.append(nn.Linear(num_neurons, 200))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(200, 20))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(20, 1))
        # self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, lower_img_output, higher_img_output, rank_difference):
        reward_diff = higher_img_output - lower_img_output
        loss = -torch.mean(rank_difference * F.logsigmoid(reward_diff))

        return loss


class Vision_Transformer(nn.Module):
    def __init__(self, pretrained_model="google/vit-large-patch32-224-in21k", reference_image=False):
        super().__init__()
        self.name = pretrained_model
        config = get_config()
        dropout = config["MODEL"]["VIT"]["DROPOUT"]
        #self.vit = ViTForImageClassification.from_pretrained(pretrained_model)

        if reference_image:
            # 1) Create a config with 6 channels
            config = ViTConfig.from_pretrained(pretrained_model)
            config.num_channels = 6

            # 2) Instantiate a new ViTModel with 6-channel config (uninitialized for patch embedding)
            self.vit = ViTModel(config)

            # 3) Load the original 3-channel model
            old_vit = ViTModel.from_pretrained(pretrained_model)

            # 4) Copy old patch embedding weights into the new 6-channel layer
            with torch.no_grad():
                # old patch embedding (3 in_channels)
                old_weight = old_vit.embeddings.patch_embeddings.projection.weight  # (embed_dim, 3, k, k)
                old_bias = old_vit.embeddings.patch_embeddings.projection.bias  # (embed_dim,)

                new_weight = self.vit.embeddings.patch_embeddings.projection.weight  # (embed_dim, 6, k, k)
                new_bias = self.vit.embeddings.patch_embeddings.projection.bias  # (embed_dim,)

                # Copy the original 3 channels
                new_weight[:, :3, :, :] = old_weight

                # copy the channels over to the new channels as well

                #new_weight[:, 3:, :, :] = old_weight

                # Random init the extra 3 channels
                nn.init.xavier_uniform_(new_weight[:, 3:, :, :])

                # Copy bias
                new_bias.copy_(old_bias)

            # 5) Load the rest of the weights from old_vit (excluding patch embedding) with strict=False
            pretrained_state = old_vit.state_dict()
            # Remove patch embedding keys so we don't overwrite our manual surgery
            del pretrained_state["embeddings.patch_embeddings.projection.weight"]
            del pretrained_state["embeddings.patch_embeddings.projection.bias"]

            self.vit.load_state_dict(pretrained_state, strict=False)

        else:
            # Just load a standard 3-channel ViT
            self.vit = ViTModel.from_pretrained(pretrained_model)

        self.hidden_size = self.vit.config.hidden_size
        clf_start = 1024
        self.vit.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, clf_start),
            #nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_start, clf_start//2),
            #nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_start//2, clf_start//4),
            #nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_start//4, 1)
        )

    def forward(self, x):
        output =self.vit(x)
        cls_emb = output.last_hidden_state[:, 0, :]
        logits = self.vit.classifier(cls_emb)
        return logits

    def pair_loss(self, lower_img_output, higher_img_output, rank_difference):
        reward_diff = higher_img_output - lower_img_output
        loss = -torch.mean(rank_difference * F.logsigmoid(reward_diff))

        return loss

    def ListNet_loss(self, predictions, true_ranks, alpha=1.0):
        exp_ranks = torch.exp(-alpha * true_ranks.float())
        p_true = exp_ranks / exp_ranks.sum()

        # Convert predicted scores to distribution P_pred via softmax
        p_pred = F.softmax(predictions, dim=0)

        # Cross-entropy
        loss = -torch.sum(p_true * torch.log(p_pred + 1e-12))  # +1e-12 to avoid log(0)
        return loss

    def list_mle_loss(self, pred_scores: torch.Tensor, true_ranks: torch.Tensor) -> torch.Tensor:
        """Computes the ListMLE loss for a single list.

        Args:
            pred_scores (torch.Tensor): shape (N,) - predicted scores for N items.
            true_ranks (torch.Tensor): shape (N,) - ground-truth ranks for these items,
                where lower rank = better (e.g., rank=1 is top item).

        Returns:
            torch.Tensor: scalar loss (negative log-likelihood).

        """
        # 1) Sequential factorization
        #    - sum_{i=1..N} [ s_{pi(i)} - log \sum_{j=i..N} exp(s_{pi(j)}) ]
        N = pred_scores.shape[0]
        nll = torch.tensor(0.0, device=pred_scores.device)
        for i in range(N):
            # partial subset from i..N-1
            subset_scores = pred_scores[i:]  # shape (N-i,)
            log_sum_exp = torch.logsumexp(subset_scores, dim=0)
            nll += (pred_scores[i] - log_sum_exp)

        # 3) Negative log-likelihood
        return -nll / N

    def neuralsort_loss(self,
                        pred_scores: torch.Tensor,
                        true_scores: torch.Tensor,
                        tau: float = 1.0) -> torch.Tensor:
        """Example loss that tries to match the model's soft-sorted pred_scores
        to the ground-truth sorted order of true_scores.

        Args:
            pred_scores: shape (N,) - predicted scores for each item
            true_scores: shape (N,) - ground-truth scores (or ranks).
                         We'll sort these to get the correct ordering.
            tau: temperature for the NeuralSort operator

        Returns:
            torch.Tensor: scalar loss

        """
        # 1) Generate the soft permutation matrix
        P = neural_sort(pred_scores, tau=tau)  # shape (N, N)

        # 2) Soft-sorted predicted scores = P^T * pred_scores
        # shape: (N,) after matrix-vector multiplication
        soft_sorted_pred = P.T @ pred_scores

        # 3) True sorted scores (lowest to highest or vice versa)
        # e.g. ascending:
        sorted_true, _ = torch.sort(true_scores, dim=0, descending=True)  # shape (N,)

        # 4) Some distance measure, e.g. L2
        loss = torch.mean((soft_sorted_pred - sorted_true) ** 2)
        return loss

    def get_correct_loss(self, loss_name):
        if loss_name == "ListNet":
            return self.ListNet_loss
        if loss_name == "ListMLE":
            return self.list_mle_loss
        if loss_name == "NeuralSort":
            return self.neuralsort_loss
        return self.pair_loss


def neural_sort(s: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Produces a soft permutation matrix (N, N) from a score vector s of shape (N,).
    Tau is the temperature controlling how 'sharp' the permutation is.

    Returns:
        P: A (N, N) tensor that is row-stochastic. P[i, :] is the distribution over
           item i's position in the sorted order.

    """
    # s shape: (N,)
    # We want a row-stochastic matrix P in R^(N x N).
    # One approach from the paper is:
    #   P = softmax( - (s - s^T)^2 / tau ), row-wise or column-wise.

    N = s.shape[0]
    # Expand s to shape (N, 1) so we can do s - s^T
    s_ = s.view(-1, 1)  # shape: (N, 1)
    # Score differences
    diff = s_ - s_.T     # shape: (N, N)

    # Or some variants use squared differences:
    #   diff_sq = -(diff**2) / tau
    #   P = softmax(diff_sq, dim=1)
    # We'll do the simpler version with negative differences:
    diff = -(diff) / tau  # shape: (N, N)
    P = F.softmax(diff, dim=1)  # row-stochastic

    return P


if __name__ == "__main__":
    # Example usage
    model = ViTModel.from_pretrained("google/vit-huge-patch14-224-in21k")
    hidden_size = model.config.hidden_size
    print(hidden_size)  # e.g. 768 for ViT-Base
