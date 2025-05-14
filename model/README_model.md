# Model Compression
- 기존 모델 성능 검증

![Validation](https://github.com/user-attachments/assets/ca2cb25e-adb5-4f64-a817-4bd6224011c4)

- 이후 경량화 진행

![Compression process](https://github.com/user-attachments/assets/1d280214-4a8e-4d6b-97a8-ee485cce77f5)

## Quantization Process
- Weight Conversion: The model weights are converted from float32 to int8 and stored as int8.
- Scale and zero point are calculated for each weight tensor.

- Example.
  + Original weight: [0.25, -0.1, 0.05, -0.3] (float32)

  + Quantized weight: [25, -10, 5, -30] (int8)

  + Scale: 0.01, Zero Point: 0

- During inference, int8 weights are dequantized back to float32 using:

```
weight_float32 = (int8_weight - zero_point) * scale
```

- The multiplication is performed as:
```python
float32_output = (weight_float32) × (input_float32)
```

## DyAD

```
model DynamicVAE(
    (encoder_rnn): GRU(7, 1024, batch_first=True, bidirectional=True)
    (decoder_rnn): GRU(4, 1024, batch_first=True, bidirectional=True)
    (hidden2mean): Linear(in_features=2048, out_features=24, bias=True)
    (hidden2log_v): Linear(in_features=2048, out_features=24, bias=True)
    (latent2hidden): Linear(in_features=24, out_features=2048, bias=True)
    (outputs2embedding): Linear(in_features=2048, out_features=3, bias=True)
    (mean2latent): Sequential(
        (0): Linear(in_features=24, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=1, bias=True)
    )
)

model size: 51.38 MB

Recall: 0.7692
False Alarm Rate (FA): 0.0000
F1 Score: 0.8696
AUC: 0.8974358974358975
```
- Dynamic Quantization

```
model DynamicVAE(
  (encoder_rnn): DynamicQuantizedGRU(7, 1024, batch_first=True, bidirectional=True)
  (decoder_rnn): DynamicQuantizedGRU(4, 1024, batch_first=True, bidirectional=True)
  (hidden2mean): DynamicQuantizedLinear(in_features=2048, out_features=24, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
  (hidden2log_v): DynamicQuantizedLinear(in_features=2048, out_features=24, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
  (latent2hidden): DynamicQuantizedLinear(in_features=24, out_features=2048, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
  (outputs2embedding): DynamicQuantizedLinear(in_features=2048, out_features=3, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
  (mean2latent): Sequential(
    (0): DynamicQuantizedLinear(in_features=24, out_features=512, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
    (1): ReLU()
    (2): DynamicQuantizedLinear(in_features=512, out_features=1, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
  )
)

model size: 12.94 MB

Recall: 0.8462
False Alarm Rate (FA): 0.0000
F1 Score: 0.9167
AUC: 0.9102564102564102
```


## Dataset

https://figshare.com/articles/dataset/Realistic_fault_detection_of_Li-ion_battery_via_dynamical_deep_learning_approach/23659323?file=41519169

## References
[1] J. Zhang et al., “Realistic fault detection of li-ion battery via dynamical deep learning,” Nature Communications, vol. 14, no. 5940, 2023.

https://github.com/962086838/Battery_fault_detection_NC_github
