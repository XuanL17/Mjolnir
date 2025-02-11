# Mjölnir: Breaking the Shield of Perturbation-Protected Gradients via Adaptive Diffusion *(AAAI-25 Main Track)*  
**Xuan Liu, Siqi Cai, Qihua Zhou, Song Guo, Ruibin Li, Kaiwei Lin**  

**Full Paper Preprint**: [https://arxiv.org/abs/2407.05285](https://arxiv.org/abs/2407.05285)
---

## **Abstract**  
Perturbation-based mechanisms, such as differential privacy, mitigate gradient leakage attacks by introducing noise into the gradients, thereby preventing attackers from reconstructing clients' private data from the leaked gradients. However, can gradient perturbation protection mechanisms truly defend against all gradient leakage attacks?  

In this paper, we present the first attempt to break the shield of gradient perturbation protection in Federated Learning for the extraction of private information. We focus on common noise distributions, specifically Gaussian and Laplace, and apply our approach to DNN and CNN models.  

We introduce **Mjölnir**, a perturbation-resilient gradient leakage attack that is capable of removing perturbations from gradients without requiring additional access to the original model structure or external data. Specifically:  
- **Diffusion-Based Gradient Denoising**: We leverage the inherent diffusion properties of gradient perturbation to develop a novel diffusion-based gradient denoising model.  
- **Surrogate Client Model**: By constructing a surrogate client model that captures the structure of perturbed gradients, we obtain crucial gradient data for training the diffusion model.  
- **Adaptive Sampling Steps**: Monitoring disturbance levels during the reverse diffusion process enhances gradient denoising capabilities, generating gradients that closely approximate the original, unperturbed versions.  

**Key Results**: Extensive experiments demonstrate that **Mjölnir** effectively recovers the protected gradients and exposes the Federated Learning process to the threat of gradient leakage, achieving superior performance in gradient denoising and private data recovery.  

---

## **Mjölnir Overview**  

### **Threat Model**  
![Threat Model](M_fig1.png)  

### **Methodology**  
![Methodology](M_fig2.png)  

---

## **Experiments**  

### **Experimental Setups**  

1. **Mjölnir Variant Attack Models**:  
   - **Mjölnir**: Trained with only unperturbed surrogate gradients.  
   - **Conditional Mjölnir**: Trained with both perturbed gradients and unperturbed surrogate gradients.  
   - **Non-Adaptive Mjölnir**: Without the adaptive process (perturbation scale $M$ is not used as an adaptive parameter during the gradient diffusion process).  

2. **Benchmarks and Datasets**:  
   - **Privacy Datasets**: MNIST, CIFAR100, and STL10 are used as client privacy datasets and serve as the ground truth for privacy leakage evaluation.  
   - **Unperturbed Gradients**: Gradients ($\nabla W$) from the local training model of the target client are used as the reference benchmark of gradient denoising under the FL-PP paradigm.  
   - **Training Dataset for Diffusion Model**: FashionMNIST gradients are used to train the Mjölnir gradient diffusion model.  

3. **Evaluation and Boundaries**:  
   - **Privacy Leakage Capability**:
     - **Image Average Peak Signal-to-Noise Ratio (PSNR)**: Measures the fidelity of recovered images.  
     - **Label Recovered Accuracy (LRA)**: Measures the accuracy of the recovered labels.  
     - **Success Definition**: The attack is successful if human visual perception can discern requisite information from recovered images.  
   - **Gradient Denoising Quality**:
     - **Cosine Similarity (CosSimilar)**: Measures the similarity between recovered gradients ($\nabla W^R$) and original gradients ($\nabla W$).  
     - **Gradient PSNR**: Measures the peak signal-to-noise ratio of recovered gradients compared to the original gradients.  

   Higher metric values indicate better recovery fidelity and gradient accuracy.  

---

## **Contact**  
For any questions or collaboration inquiries, please feel free to contact:  
- **Xuan Liu**: [xuan18.liu@polyu.edu.hk](mailto:xuan18.liu@polyu.edu.hk)  
- **Siqi Cai**: [csiqi@whut.edu.cn](mailto:csiqi@whut.edu.cn)

