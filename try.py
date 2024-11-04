import torch


def is_mps_available():
    # Check if MPS is available on your machine
    if not hasattr(torch.backends, "mps"):
        print("MPS backend is not available in this version of PyTorch.")
        return False
    if not torch.backends.mps.is_available():
        print("MPS is not available.")
        return False

    print("MPS is available.")
    return True


def test_mps_tensor_operations():
    try:
        # Create a tensor
        x = torch.tensor([1.0, 2.0, 3.0], device="mps")
        y = torch.tensor([4.0, 5.0, 6.0], device="mps")

        # Perform some operations
        z = x + y
        print(f"Tensor x: {x}")
        print(f"Tensor y: {y}")
        print(f"Tensor z (x + y): {z}")

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    if is_mps_available():
        test_success = test_mps_tensor_operations()
        if test_success:
            print("MPS tensor operations are working correctly.")
        else:
            print("MPS tensor operations failed.")
