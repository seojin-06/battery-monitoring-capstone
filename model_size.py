import os
import torch
import argparse
from model import dataset
from model import tasks
from model import dynamic_vae

def parse_args():
    parser = argparse.ArgumentParser(description="Load and display model information.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model folder")
    return parser.parse_args()


def print_model_size(mdl):
    """
    모델 크기를 MB 단위로 출력하는 함수
    """
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
    os.remove('tmp.pt')


def main():
    args = parse_args()
    
    model_file = os.path.join(args.model_path, "model.torch")
    
    if not os.path.exists(model_file):
        print(f"Model file not found at {model_file}")
        return
    
    model = torch.load(model_file)

    # 모델 크기 확인
    print_model_size(model)

    # 파라미터 수 확인
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    print(model.encoder_rnn.weight_ih_l0.dtype)  # 가중치 타입 확인

    # 각 파라미터의 층 이름 출력
    print("\nModel Layers:")
    for name, _ in model.named_parameters():
        print(name)

    # 각 파라미터의 값 출력
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")


if __name__ == '__main__':
    main()
