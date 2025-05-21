apt-get install -y build-essential

mkdir /arm64 && cd /arm64
uv venv --python 3.12
source .venv/bin/activate
uv pip install setuptools wheel numpy

uv pip install "neo4j==5.28.1" "typing-extensions==4.13.0" "sympy==1.13.1" "scipy==1.15.2" "dotenv==0.9.9" "scikit-learn==1.6.1"
uv pip install "torch==2.5.0" --index-url https://download.pytorch.org/whl/cpu

# Now install PyTorch Geometric and extensions
uv pip install --no-build-isolation "torch-geometric==2.5.0"

uv pip install --no-build-isolation --verbose "git+https://github.com/rusty1s/pytorch_sparse.git@0.6.18"
uv pip install --no-build-isolation --verbose "git+https://github.com/rusty1s/pytorch_scatter.git@2.1.2"

python3 -c "
import torch
import torch_geometric
import torch_scatter
import torch_sparse

print('PyTorch version:', torch.__version__)
print('PyTorch Geometric version:', torch_geometric.__version__)
print('Torch Scatter version:', torch_scatter.__version__)
print('Torch Sparse version:', torch_sparse.__version__)

# Test scatter
src = torch.randn(5, 3)
index = torch.tensor([0, 1, 0, 1, 2])
out = torch_scatter.scatter_mean(src, index, dim=0)
print('Scatter mean output shape:', out.shape)

# Test sparse
indices = torch.tensor([[0, 1], [1, 0]])
values = torch.tensor([1.0, 2.0])
sparse = torch.sparse_coo_tensor(indices, values, (2, 2))
print('Sparse tensor to dense:', sparse.to_dense())
"
