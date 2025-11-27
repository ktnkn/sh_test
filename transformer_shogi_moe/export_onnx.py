import torch
import torch.onnx
import argparse
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformer_shogi_moe.model import TransformerPolicyValueNetwork
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi import serializers

def convert_to_fp16(onnx_path, output_path):
    """Convert ONNX model to FP16"""
    try:
        from onnxconverter_common import float16
        import onnx
        
        print(f"Converting {onnx_path} to FP16...")
        model = onnx.load(onnx_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, output_path)
        print(f"FP16 model saved to {output_path}")
        return True
    except ImportError:
        print("Warning: onnxconverter-common not installed. Install with: pip install onnxconverter-common")
        print("Skipping FP16 conversion.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Export Transformer Shogi model to ONNX')
    parser.add_argument('model', type=str, help='Input model file path (.pth or .npz)')
    parser.add_argument('output', type=str, help='Output ONNX file path')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_attention_head', type=int, default=8)
    parser.add_argument('--n_kv_head', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--mtp_heads', type=int, default=0)
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--num_experts_per_tok', type=int, default=1)
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--fp16', action='store_true', help='Convert to FP16 (requires onnxconverter-common)')
    
    args = parser.parse_args()

    device = torch.device("cpu")
    
    print(f"Loading model from {args.model}...")
    model = TransformerPolicyValueNetwork(
        d_model=args.d_model,
        n_attention_head=args.n_attention_head,
        n_kv_head=args.n_kv_head,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.0,
        mtp_heads=args.mtp_heads,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok
    )
    model.eval()
    model.to(device)

    if args.model.endswith('.pth'):
        checkpoint = torch.load(args.model, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        serializers.load_npz(args.model, model)

    # Dummy input
    batch_size = 1
    x1 = torch.zeros(batch_size, FEATURES1_NUM, 9, 9, dtype=torch.float32, device=device)
    x2 = torch.zeros(batch_size, FEATURES2_NUM, 9, 9, dtype=torch.float32, device=device)

    # Determine output path
    if args.fp16:
        temp_output = args.output.replace('.onnx', '_fp32.onnx') if args.output.endswith('.onnx') else args.output + '_fp32'
    else:
        temp_output = args.output

    output_names = ['output_policy', 'output_value']
    
    # Wrapper to strip aux_loss and flatten MTP outputs
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model, mtp_heads):
            super().__init__()
            self.model = model
            self.mtp_heads = mtp_heads
            
        def forward(self, x1, x2):
            if self.mtp_heads > 0:
                y1, y2, mp, mv, aux_loss = self.model(x1, x2)
                return (y1, y2, *mp, *mv)
            else:
                y1, y2, aux_loss = self.model(x1, x2)
                return y1, y2

    model = ONNXWrapper(model, args.mtp_heads)

    if args.mtp_heads > 0:
        for i in range(args.mtp_heads):
            output_names.append(f'mtp_policy_{i}')
        for i in range(args.mtp_heads):
            output_names.append(f'mtp_value_{i}')

    print(f"Exporting to {temp_output}...")
    torch.onnx.export(
        model,
        (x1, x2),
        temp_output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input1', 'input2'],
        output_names=output_names,
        dynamic_axes={
            'input1': {0: 'batch_size'},
            'input2': {0: 'batch_size'},
            'output_policy': {0: 'batch_size'},
            'output_value': {0: 'batch_size'}
        }
    )
    print("Export completed.")

    # Convert to FP16 if requested
    if args.fp16:
        if convert_to_fp16(temp_output, args.output):
            # Remove temporary FP32 file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                print(f"Removed temporary file: {temp_output}")

if __name__ == '__main__':
    main()
