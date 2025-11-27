import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformer_shogi_moe.model import TransformerPolicyValueNetwork


def count_parameters(model):
    """モデルの総パラメータ数を計算"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """数値を読みやすい形式にフォーマット"""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def main():
    parser = argparse.ArgumentParser(description='モデルのパラメータ数を計算')
    parser.add_argument('--d_model', type=int, default=512, help='モデルの次元数')
    parser.add_argument('--n_attention_head', type=int, default=8, help='アテンションヘッド数')
    parser.add_argument('--n_kv_head', type=int, default=2, help='KVヘッド数')
    parser.add_argument('--num_layers', type=int, default=16, help='レイヤー数')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='FFN中間層の次元数')
    parser.add_argument('--mtp_heads', type=int, default=0, help='MTPヘッド数')
    parser.add_argument('--num_experts', type=int, default=2, help='エキスパート数')
    parser.add_argument('--num_experts_per_tok', type=int, default=1, help='トークンごとのエキスパート数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("モデル設定:")
    print("=" * 60)
    print(f"d_model:              {args.d_model}")
    print(f"n_attention_head:     {args.n_attention_head}")
    print(f"n_kv_head:            {args.n_kv_head}")
    print(f"num_layers:           {args.num_layers}")
    print(f"dim_feedforward:      {args.dim_feedforward}")
    print(f"mtp_heads:            {args.mtp_heads}")
    print(f"num_experts:          {args.num_experts}")
    print(f"num_experts_per_tok:  {args.num_experts_per_tok}")
    print("=" * 60)
    
    # モデルを作成
    model = TransformerPolicyValueNetwork(
        d_model=args.d_model,
        n_attention_head=args.n_attention_head,
        n_kv_head=args.n_kv_head,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.1,
        mtp_heads=args.mtp_heads,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok
    )
    
    # パラメータ数を計算
    total_params, trainable_params = count_parameters(model)
    

    
    # 各コンポーネントのパラメータ数を表示
    print("\nコンポーネント別パラメータ数:")
    print("=" * 60)
    
    component_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        component_params[name] = params
        print(f"{name:20s}: {params:>12,} ({format_number(params)})")
    
    print("=" * 60)
    
    # レイヤーごとの詳細（最初のレイヤーのみ）
    if hasattr(model, 'layers') and len(model.layers) > 0:
        print("\nレイヤー詳細（1層あたり）:")
        print("=" * 60)
        layer = model.layers[0]
        for name, module in layer.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"{name:20s}: {params:>12,} ({format_number(params)})")
        
        layer_params = sum(p.numel() for p in layer.parameters())
        print("-" * 60)
        print(f"{'1層の合計':20s}: {layer_params:>12,} ({format_number(layer_params)})")
        print(f"{'全レイヤーの合計':20s}: {layer_params * args.num_layers:>12,} ({format_number(layer_params * args.num_layers)})")
        print("=" * 60)

    print("\nパラメータ数:")
    print("=" * 60)
    print(f"総パラメータ数:       {total_params:,} ({format_number(total_params)})")
    print(f"学習可能パラメータ数: {trainable_params:,} ({format_number(trainable_params)})")
    print("=" * 60)

if __name__ == '__main__':
    main()
