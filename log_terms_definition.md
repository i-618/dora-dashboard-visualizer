# Audiocraft Watermark Model Training Log Terms

---

## Four Most Important Metrics for Watermark Model Training

To monitor the health and efficacy of the Watermark model during training, focus on these four categories of metrics:

1.  **Generator Loss (`g_loss`):** This must trend **downwards**. It indicates the generator's success in creating high-quality, watermarked audio that fools the discriminator while maintaining perceptual quality and successfully embedding the watermark.
2.  **Watermark Detection Identity (`wm_detection_identity`):** This should be **high (near 1.0 or 100%)** and **stable** (in this case it should remain **low/stable** if reported as loss) . It measures the detector's ability to reliably find the watermark in *undistorted* watermarked audio. This is the baseline success metric.
3.  **Robustness Message Bit Error Rate (`wm_mb_X` terms, e.g., `wm_mb_identity`, `wm_mb_mp3_compression`):** This should be **low (near 0.0)**. It measures the message decoding error after common attacks like MP3 compression or filtering. A low value means the embedded message is robustly recoverable.
4.  **Audio Quality Metrics (`pesq`, `sisnr`):** These should remain **high** or **improve**. These metrics ensure the watermark is **imperceptible** and is not degrading the listener's experience.

---

## I. Introduction to Audiocraft and Audio Watermarking

**Audiocraft** represents a sophisticated PyTorch library developed by Facebook Research, dedicated to advancing deep learning in audio processing and generation. [1, 2] This comprehensive framework provides both inference and training capabilities for state-of-the-art AI generative models. Notable examples include **MusicGen**, **AudioGen**, and **EnCodec**. [1, 3]

The library's modularity is evident in its support for watermarking, often tied to the **AudioSeal** model. [1, 3] The training logs track a multitude of loss functions and metrics that collectively reflect the efficacy of watermark embedding, detection accuracy, and resilience against various audio distortions, all while striving to maintain the original audio's **perceptual quality**.

The core challenge is a **multi-objective optimization problem**: the watermark must be **imperceptible**, yet **robust** enough to withstand attacks, and **reliably detectable**.

---

## II. General Training and System Metrics

| Term | Definition | Reference |
| :--- | :--- | :--- |
| **Epoch** | Represents the smallest quantum of training iterations before a **checkpoint** is initiated, rather than a strict full pass over the entire dataset. This is a pragmatic definition for managing large-scale, distributed training. | [4] |
| **max\_mem** | Quantifies the **peak GPU memory consumption** (typically in Gigabytes) occupied by tensors on a given device during a training step. High values signal inefficient memory use or potential out-of-memory errors. | [6, 7, 8] |
| **duration** | The elapsed time for processing audio segments or completing specific training or evaluation steps, often converted to **minutes** in the logs for readability. | [9] |

---

## III. Core Loss Functions

These losses are the primary drivers of the model's learning and convergence.

| Term | Definition | Reference |
| :--- | :--- | :--- |
| **d\_loss** | **Discriminator Loss**. Measures how effectively the discriminator distinguishes between watermarked ("fake") audio and original ("real") clean audio. The discriminator is trained to **maximize** this loss. | [8] |
| **g\_loss** | **Generator Loss**. The aggregate loss the generator (watermark embedder) strives to **minimize**. It represents the generator's success in producing high-fidelity watermarked audio that **fools the discriminator**. | [8] |
| **l1** | **L1 (Mean Absolute Error) loss** on the **MelSpectrogram** representation of the audio signals. Minimizing this ensures the watermarked audio's spectral characteristics remain close to the original, crucial for **imperceptibility**. | [10] |
| **msspec** | **Multi-Scale MelSpectrogram Loss**. A composite loss calculated across **multiple frequency scales** of MelSpectrograms to capture differences at various frequency granularities. Used to encourage perceptually accurate audio generation. | [10, 11] |
| **adv** | **Adversarial Loss** component for the **Generator**. Quantifies the generator's success in tricking the discriminator based on its final classification output. | [12] |
| **feat** | **Feature Matching Loss** component for the **Generator**. Encourages the intermediate feature maps extracted by the discriminator from the watermarked audio to closely resemble those from the clean audio, which helps **stabilize training** and increase robustness. | [12] |
| **tf\_loudnessratio** | **Time-Frequency Loudness Ratio Loss**. Quantifies the difference in **perceived loudness** between the watermarked and clean signals. Minimizing this is vital for ensuring the watermark does not introduce noticeable volume changes. | [13] |

---

## IV. Discriminator-Specific Outputs (MS-STFT Discriminator)

These terms relate to the **`MultiScaleSTFTDiscriminator`**, which analyzes the audio at multiple frequency and temporal resolutions to enhance watermark robustness.

| Term | Definition | Reference |
| :--- | :--- | :--- |
| **d\_msstftd** | **Discriminator Loss** specifically from the `MultiScaleSTFTDiscriminator`. | [14, 15] |
| **adv\_msstftd** | **Adversarial Loss** for the Generator, as evaluated by the `MultiScaleSTFTDiscriminator`. | [14, 15] |
| **feat\_msstftd** | **Feature Matching Loss** for the Generator, using feature maps from the `MultiScaleSTFTDiscriminator`. | [14, 15] |

---

## V. Quality Evaluation Metrics

These objective metrics evaluate the perceptual fidelity of the watermarked audio.

| Term | Definition | Reference |
| :--- | :--- | :--- |
| **pesq** | **Perceptual Evaluation of Speech Quality**. An objective metric where a **higher score** indicates superior perceived audio quality, signifying less degradation from the original. A high PESQ confirms **imperceptibility**. | [8, 16] |
| **mel** | Refers to the **Mel-frequency scale**, which corresponds to **human auditory perception**. Metrics based on this scale (like `l1` and `msspec`) are used to ensure perceptual similarity. | [10, 17, 18] |
| **sisnr** | **Scale-Invariant Signal-to-Noise Ratio**. A robust metric to quantify distortion or noise. Audiocraft may report the negative SI-SNR (a loss function) where **lower values** signify better fidelity. | [19, 20, 22] |

---

## VI. Optimization and Training Ratios

These ratios provide insight into the stability and balance of the optimization process, especially how the various loss components are weighted.

| Term | Definition | Reference |
| :--- | :--- | :--- |
| **ratio1** | Calculated as the L2 norm of the gradients **after** the `wm_detection_` and `wm_mb_` losses are applied, but **before** the main adversarial losses are incorporated by the `balancer`. Monitors early gradient magnitude. | [8, 23] |
| **ratio2** | Calculated as the L2 norm of the gradients **after** the `self.balancer.backward` call. Reflects the **aggregate gradient magnitude** from all loss components, indicating overall **training stability**. | [8, 23] |

---

## VII. Watermark Robustness Metrics (Augmentation-Specific)

These terms are critical for measuring the watermark's ability to survive real-world audio processing. The model is trained against these augmentations to ensure **robustness**.

- **`wm_detection_X`**: A loss component that enforces the detector's ability to correctly identify the presence of the watermark in the **augmented watermarked audio**. Lower loss implies better detection robustness.
- **`wm_mb_X`**: A loss component that enforces the accurate recovery of the **message bits** ("mb") from the **augmented watermarked audio**. Lower loss implies better message recovery robustness.

| Log Key | Associated Audio Transformation | Purpose / Definition | Reference |
| :--- | :--- | :--- | :--- |
| **wm\_detection\_identity** | No Transformation (Identity) | Baseline detection loss for ideal conditions. | [8] |
| **wm\_mb\_identity** | No Transformation (Identity) | Baseline message bit error loss for ideal conditions. | [8] |
| **wm\_detection\_mp3\_compression** | MP3 Compression | Measures detector's ability to detect watermark after MP3 compression. | [16, 27] |
| **wm\_mb\_mp3\_compression** | MP3 Compression | Measures message bit accuracy after MP3 compression. | [16, 27] |
| **wm\_detection\_aac\_compression** | AAC Compression | Measures detection robustness after AAC compression. | [16, 27] |
| **wm\_mb\_aac\_compression** | AAC Compression | Measures message bit accuracy after AAC compression. | [16, 27] |
| **wm\_detection\_encodec\_nq** | EnCodec Neural Quantization | Measures robustness after neural audio compression/quantization. | [25] |
| **wm\_mb\_encodec\_nq** | EnCodec Neural Quantization | Measures message bit accuracy after neural quantization. | [25] |
| **wm\_detection\_lowpass\_filter** | Low-Pass Filtering | Measures robustness after low-pass frequency filtering. | [16, 25] |
| **wm\_mb\_lowpass\_filter** | Low-Pass Filtering | Measures message bit accuracy after low-pass frequency filtering. | [16, 25] |
| **wm\_detection\_bandpass\_filter** | Band-Pass Filtering | Measures robustness after band-pass frequency filtering. | [16, 25] |
| **wm\_mb\_bandpass\_filter** | Band-Pass Filtering | Measures message bit accuracy after band-pass filtering. | [16, 25] |
| **wm\_detection\_highpass\_filter** | High-Pass Filtering | Measures robustness after high-pass frequency filtering. | [16, 25] |
| **wm\_mb\_highpass\_filter** | High-Pass Filtering | Measures message bit accuracy after high-pass filtering. | [16, 25] |
| **wm\_detection\_pink\_noise** | Pink Noise Addition | Measures robustness in the presence of pink noise. | [16] |
| **wm\_mb\_pink\_noise** | Pink Noise Addition | Measures message bit accuracy in the presence of pink noise. | [16] |
| **wm\_detection\_updownresample** | Up/Down Resampling | Measures robustness after changing the sample rate. | [25] |
| **wm\_mb\_updownresample** | Up/Down Resampling | Measures message bit accuracy after resampling. | [25] |
| **wm\_detection\_boost\_audio** | Audio Boosting/Gain | Measures robustness after audio amplitude is boosted. | [8, 25] |
| **wm\_mb\_boost\_audio** | Audio Boosting/Gain | Measures message bit accuracy after audio amplitude is boosted. | [8, 25] |
| **wm\_detection\_duck\_audio** | Audio Ducking | Measures robustness after dynamic volume reduction. | [25] |
| **wm\_mb\_duck\_audio** | Audio Ducking | Measures message bit accuracy after dynamic volume reduction. | [25] |
| **wm\_detection\_speed** | Speed Change/Time Stretch | Measures robustness after speed perturbation or time stretching. | [16, 24, 25] |
| **wm\_mb\_speed** | Speed Change/Time Stretch | Measures message bit accuracy after speed perturbation or time stretching. | [16, 24, 25] |
| **wm\_detection\_echo** | Echo Effect | Measures robustness after an echo effect is applied. | [16, 25] |
| **wm\_mb\_echo** | Echo Effect | Measures message bit accuracy after an echo effect is applied. | [16, 25] |
| **wm\_detection\_smooth** | Smoothing/Blurring | **DON'T KNOW** - Specific implementation details for 'smooth' not explicitly listed in typical augmentations; likely a time-domain filtering operation. | DON'T KNOW |
| **wm\_mb\_smooth** | Smoothing/Blurring | **DON'T KNOW** - Specific implementation details for 'smooth' not explicitly listed in typical augmentations; likely a time-domain filtering operation. | DON'T KNOW |
| **duration** | NA |The elapsed time for processing audio segments or completing specific training or evaluation steps. (Repeated for completeness). | [9] |

---

## VIII. References

The definitions and references are derived from the structure and content of the `facebookresearch/audiocraft` GitHub repository, specifically within the following files:

| Ref | Source File/Context | Description |
| :--- | :--- | :--- |
| [1] | `audiocraft/watermark/README.md` | General context, AudioSeal identity, installation details. |
| [2] | `facebookresearch/audiocraft` (Repo) | General model context (MusicGen, AudioGen). |
| [3] | `audiocraft/docs/WATERMARK.md` | General watermarking context and model identity. |
| [4] | `audiocraft/modules/checkpoint.py` | Custom definition of "Epoch" related to checkpointing strategy. |
| [5] | General ML Knowledge | Conventional definition of an epoch. |
| [6] | `audiocraft/utils/throughput.py` | `max_mem` calculation using `torch.cuda.max_memory_allocated`. |
| [7] | PyTorch Documentation | General source for `max_mem` (maximum memory allocated). |
| [8] | `audiocraft/solvers/watermark.py` | Primary source for training loop, loss calculation (`d_loss`, `g_loss`, `ratio1`, `ratio2`, `pesq`), and augmentation application. |
| [9] | `audiocraft/solvers/explorer.py` | Conversion of `duration` (in seconds) to minutes for logging. |
| [10] | `audiocraft/losses/specloss.py` | Implementation of `MelSpectrogramL1Loss` (`l1`) and `MultiScaleMelSpectrogramLoss` (`msspec`). |
| [11] | `audiocraft/losses/specloss.py` | Detailed structure of `MultiScaleMelSpectrogramLoss`. |
| [12] | `audiocraft/adversarial/losses.py` | Implementation of `AdversarialLoss` and `FeatureMatchingLoss` (`adv` and `feat`). |
| [13] | `audiocraft/losses/misc.py` | Implementation of `TFLoudnessRatio` (`tf_loudnessratio`). |
| [14] | `audiocraft/modules/discriminator.py` | Architecture of `MultiScaleSTFTDiscriminator`. |
| [15] | `audiocraft/solvers/watermark.py` | Usage of the MS-STFT Discriminator components in the solver. |
| [16] | AudioSeal Research Context | General knowledge/documentation on common watermarking robustness tests (MP3, Noise, Filtering). |
| [17] | General Audio Processing | Mel-frequency scale definition. |
| [18] | Speech Signal Processing | Purpose of log Mel spectrograms. |
| [19] | SI-SNR Documentation | Definition of Scale-Invariant Signal-to-Noise Ratio. |
| [20] | `audiocraft/metrics/misc.py` | Implementation of `SISNR` and its dual role as a loss function (sign flip). |
| [22] | `audiocraft/metrics/misc.py` | Context for `SISNR` metric. |
| [23] | PyTorch/Optimization Context | General context for using L2 norm of gradients (`ratio1`, `ratio2`). |
| [24] | `audiocraft/modules/augmentation.py` | Implementation of `Speed` augmentation (time stretch). |
| [25] | `audiocraft/modules/augmentation.py` | Implementation details for various augmentations (`boost_audio`, `duck_audio`, `encodec_nq`, `updownresample`, `lowpass_filter`, `echo`, `bandpass_filter`, `highpass_filter`). |
| [27] | `audiocraft/modules/augmentation.py` | Implementation details for `LossyCompression` (MP3/AAC). |
