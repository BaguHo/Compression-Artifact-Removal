import itertools
import subprocess

# 튜닝할 파라미터 후보
embed_dims = [12, 24, 48]
num_heads = [12]
num_layers = [12, 16]
mlp_dims = [48, 96, 192]
learning_rates = [0.001, 0.0005]

# 모든 조합 생성
param_grid = list(itertools.product(embed_dims, num_heads, num_layers, mlp_dims, learning_rates))

for embed_dim, num_head, num_layer, mlp_dim, lr in param_grid:
    # 실험 이름 지정
    exp_name = f"ed{embed_dim}_nh{num_head}_nl{num_layer}_mlp{mlp_dim}_lr{lr}"
    # subprocess로 PxT_y.py 실행 (필요시 파라미터를 인자로 넘기도록 PxT_y.py 수정 필요)
    cmd = [
        "python", "code/PxT_y_v2.py",
        "--epochs", "10",
        "--batch_size", "2048",
        "--num_workers", "64",
        "--img_size", "8",
        "--patch_size", "1",
        "--in_channels", "1",
        "--embed_dim", str(embed_dim),
        "--num_heads", str(num_head),
        "--num_layers", str(num_layer),
        "--mlp_dim", str(mlp_dim),
        "--lr", str(lr),
        "--model_name", exp_name
    ]
    subprocess.run(cmd)