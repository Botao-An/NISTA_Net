# NISTA-Net

1. We provide the main code for paper "***Interpretable Neural Network via Algorithm Unrolling for Mechanical Fault Diagnosis***". 

   

2. **Note**: We made some minor revisions to NISTA-Net proposed in the above paper, so the results may be not exactly the same. But it does not change the main conclusions of this paper.

   

3. The code is originally debugged on the computer with the following configuration.

   |          Hardware           |    Software    |     Software      |
   | :-------------------------: | :------------: | :---------------: |
   |  Intel Core i7-10700KF CUP  | Anaconda 4.9.2 |     CUDA 11.7     |
   |          RAM 32GB           |   Python 3.8   |    cuDNN 8.4.0    |
   | NVIDIA GeForce RTX 3080 GPU | PyTorch 1.8.1  | TorchVision 0.9.1 |



3. We provide a simulation demo for you to check the performance of NISTA-Net initially.

   

4. To run the model, you should firstly prepare the dataset in a standard format as the simulated one. More details can be found in file **train_and_postprocess.ipynb**. 

   

5. If you are interested in algorithm unrolling and would like to do reasearch based on this code, please cite the following papers. For any questions you can contact e-mail: Albert_An@foxmail.com or wsa17131026@stu.xjtu.edu.cn. 

   ```latex
   @article{an2022interpretable,
   	title={Interpretable Neural Network via Algorithm Unrolling for Mechanical Fault Diagnosis},
   	author={An, Botao and Wang, Shibin and Zhao, Zhibin and Qin, Fuhua and Yan, Ruqiang and Chen, Xuefeng},
   	journal={IEEE Transactions on Instrumentation and Measurement},
   	volume={71},
   	pages={1--11},
   	year={2022},
   	publisher={IEEE}
   }
   @article{an2022adversarial,
      title={Adversarial Algorithm Unrolling Network for Interpretable Mechanical Anomaly Detection},
      author={An, Botao and Wang, Shibin and Qin, Fuhua and Zhao, Zhibin and Yan, Ruqiang and Chen, Xuefeng},
      journal={IEEE Transactions on Neural Networks and Learning Systems},
      volume={},
      pages={1--14},
      year={2023},
      publisher={IEEE}
   }
   ```

   

6. Copyright reserved by the authors

