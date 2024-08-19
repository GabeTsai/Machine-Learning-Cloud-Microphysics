import torch 
import DeepMLPModel

def testDeepMLP():
    """
    Make sure forward pass of DeepMLP model works
    """

    input_dim = 3
    latent_dim = 64
    output_dim = 1
    output_bias = torch.tensor([1.0])
    num_blocks = 3

    model = DeepMLPModel.DeepMLP(input_dim, latent_dim, output_dim, output_bias, num_blocks)
    data = torch.randn(5, input_dim)
    out = model(data)
    print(out)
    assert out.shape == (5, output_dim), f"Expected output shape (5, {output_dim}), got {out.shape}"
    print('DeepMLP test passed')

def main():
    testDeepMLP()

if __name__ == "__main__":
    main()
