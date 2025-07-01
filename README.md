# DCGAN on CelebA
This project was inspired by my exploration of Sora and my curiosity about how images are generated using artificial intelligence. After learning about Generative Adversarial Networks (GANs), I decided to start with a simpler implementation rather than the more advanced StyleGAN, in order to better understand the fundamentals of image generation. My goal was to gain hands-on experience with GANs and observe the kinds of images I could create using a basic model.

This repository documents my effort to train a Deep Convolutional Generative Adversarial Network (DCGAN) on the CelebA dataset using PyTorch.

## 📌 Project Overview

- **Goal**: Generate realistic human face images using a DCGAN trained on CelebA.
- **Dataset**: [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) (cropped and resized to 64×64 resolution)
- **Framework**: PyTorch

## ⚙️ Setup

```bash
# install dependencies
pip install torch torchvision matplotlib numpy
```
---

## 🤖 What is a DCGAN?

A **DCGAN (Deep Convolutional GAN)** combines the idea of Generative Adversarial Networks (GANs) with convolutional neural networks (CNNs).  
It consists of two models:

- **Generator (G)**: Takes random noise as input and generates fake images.
- **Discriminator (D)**: Tries to distinguish between real images (from the dataset) and fake images (from the generator).

The models are trained adversarially:  
G tries to fool D, and D tries to correctly classify real vs fake.

### DCGAN Architecture

Below is a simplified diagram of a DCGAN:

**Generator:**

![Screenshot 2025-07-01 183758](https://github.com/user-attachments/assets/66ae34ad-671f-4ff7-aea3-df51301fd7bb)

**Discriminator:**

![Screenshot 2025-07-01 183725](https://github.com/user-attachments/assets/5497cf92-bfb5-4d2d-ab2e-010f1f2891e0)


### How does it work ?
A DCGAN works by taking random noise as input and passing it through the generator, which upsamples the noise to create a fake image. This fake image, along with real images from the dataset, is fed into the discriminator. The discriminator tries to distinguish between real and fake images and calculates a loss. This loss guides both networks: the discriminator learns to get better at telling real from fake, while the generator uses the feedback to produce more realistic images over time.

---

## 🏗 Model Details
* **Generator:** DCGAN-style transposed convolution layers with BatchNorm and ReLU activations.
* **Discriminator:** Convolutional layers with BatchNorm, Dropouts and LeakyReLU activations.
* **Loss:** Binary Cross Entropy
* **Optimizer:** Adam (lr=0.0002, betas=(0.5, 0.999))

## 🧑‍💻 Training Summary

* **Dataset:** CelebA (64×64)
* **Batch size:** Tried various values (initially large, later halved to improve results)
* **Number of epochs:** Trained for a total of 30 epochs (including restarts and adjustments)
* **Experiments:** 
  * Adjusted batch size
  * Tuned learning rate
  * Modified training duration
  * Restarted and resumed from checkpoints
  * Carefully monitored losses and visual outputs
## 📈 Results
* ✅ CIFAR-10: Achieved reasonable generated images during earlier experiments, but unfortunately forgot to download/save them before switching to CelebA.
* ⚠️ CelebA: After many changes (batch size, learning rate, etc.), could not achieve satisfactory face generations. Images often showed noisy blobs or incomplete features.

Generated samples from CelebA training are available in the 'results/' folder.
(Note:These show intermediate stages — model did not converge to desired quality within the training time. If you'd like to see the progression or blobs/noise stages, you can check these out.)

## 🚧 Issues Faced and Learnings
Initially, I trained the DCGAN on CelebA with a batch size of 128 and experimented with various adjustments—modifying learning rates for both the generator and discriminator, applying dropout, implementing label smoothing, and giving extra training passes to the generator. Despite these efforts, the generated images mostly remained noisy and lacked recognizable features.

After reducing the batch size to 64, I observed the emergence of more coherent blobs in the outputs, though images still did not reach the desired quality. Through this process, I learned several valuable lessons:

* DCGANs on CelebA require longer training (often 100+ epochs) for good results. The model may not produce realistic faces within the first few dozen epochs.
* Smaller batch sizes can stabilize training but slow down convergence. Reducing the batch size helped in achieving more stable outputs, though it also increased the time needed for training.
* Hyperparameter tuning and possibly architectural improvements (e.g., WGAN-GP, StyleGAN) could help. While my current results are limited, further experimentation with advanced architectures and hyperparameters may lead to better outcomes.

These insights will guide my future experiments and improvements to the model.

## 🎯 Things I Thought of Trying

After successful replication, I had planned to explore:

* Generating faces with **No hair on head**
* Generating faces with **No eyes**
* Generating faces with **No mouth**
* ...and other fun experiments to creatively manipulate facial features 😁  

This idea was inspired by techniques from the official DCGAN paper (e.g., removing windows in generated room images). You can achieve these kinds of modifications by:

* Tracking the **feature maps** across epochs and observing which weight value ranges correspond to specific features.
* Adjusting or zeroing out these weights (e.g., for removing eyes or hair) — but only in the **later epochs**, so that only high-level features are altered.  
* Avoid modifying these values in the early stages, as it can disrupt training or cause the network to relearn those features.  

👉 For detailed guidance on such manipulations, refer to the **[Official DCGAN Paper](https://arxiv.org/abs/1511.06434)**.

## 🚀 Future Work
* Consider advanced GAN variants (e.g., WGAN-GP, StyleGAN) for more stable training.
* Try training for 100+ epochs, possibly using gradient penalty or spectral normalization.
* Explore using different datasets or resolutions.

## 📚 References

* [Radford et al., DCGAN Paper (2015)](https://arxiv.org/abs/1511.06434)
* [GeeksForGeeks DCGAN tutorial](https://www.geeksforgeeks.org/machine-learning/deep-convolutional-gan-with-keras/)

## 🤝 Got Suggestions or Improvements?

If you have ideas to improve this project, managed to get better results, or just want to connect — feel free to reach out!  

📧 **Email:** [bansalvidhan66@gmail.com]  
💼 **LinkedIn:** [Vidhan Bansal](https://www.linkedin.com/in/vidhan-bansal-9bb784290/)  

I'd love to hear from you! 🚀
