import numpy as np
from collections import defaultdict

class SimplePartSegEvaluator:
    def __init__(self):
        # Accumulators per part_id
        self.metrics_per_part = defaultdict(list)

    @staticmethod
    def compute_iou(pred_mask, gt_mask):
        pred_mask = pred_mask.astype(bool)
        gt_mask = gt_mask.astype(bool)

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = intersection / union if union != 0 else 0.0
        return iou

    def process(self, data):
        part_id = int(data['part_id'].item())  # Extract part_id
        gt_mask = data['gt_mask'].cpu().numpy()  # [H, W] binary
        pred_mask = data['pred_mask'].cpu().numpy()  # [H, W] binary

        iou = self.compute_iou(pred_mask, gt_mask)
        self.metrics_per_part[part_id].append(iou)

    def evaluate(self):
        print("\nPer-Part Evaluation (mIoU):\n")
        all_ious = []

        for part_id in sorted(self.metrics_per_part.keys()):
            ious = self.metrics_per_part[part_id]
            mean_iou = np.mean(ious)
            print(f"Part ID {part_id:3d}: mIoU = {mean_iou:.4f} (n={len(ious)})")
            all_ious.append(mean_iou)

        overall_miou = np.mean(all_ious)
        print(f"\nOverall mIoU across parts: {overall_miou:.4f}")
        return self.metrics_per_part
