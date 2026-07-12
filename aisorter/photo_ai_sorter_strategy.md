# Ultra-Lightweight Photo AI Sorter: Folder-Based Strategy

This document outlines the strategic architecture designed to automatically classify, tag, and sort a photo library using a custom, resource-optimized pipeline. It is intentionally optimized for execution on constrained x86 hardware—specifically a 1st-generation Intel Core i7 NAS processor lacking modern vector instructions like AVX/AVX2.

---

## 1. Core Architecture Strategy

Instead of deploying heavy, monolithic multi-class networks or computationally expensive segmentation frameworks, the pipeline decouples tasks into **specialized micro-models**. 

The system relies on a **Cascading Dependency Pipeline** alongside a **Folder-Based Multi-Label Strategy** to minimize hardware utilization.

### The Pipeline Flow
1. **Step 1: Face Detection** (Ultra-Lightweight ~1MB SSD model)
   * Scans the image at low resolution (320 × 240).
   * Returns pixel boundary boxes for any human faces found.
2. **Step 2: Face Feature Extraction** (MobileFaceNet ~4MB)
   * **Conditional Execution:** This step *only* wakes up if Step 1 detects a face.
   * Converts the cropped face into a 128-dimensional mathematical vector.
   * Computes the Cosine or Euclidean distance against reference face vectors to assign names (e.g., `"John"`).
3. **Step 3: General Category Sorter** (Custom CNN < 250k parameters)
   * Evaluates the image against broad organizational buckets: `[Car, Person, Scenery]`.
   * Operates as a **Multi-Label** classifier, answering multiple independent binary questions simultaneously.
4. **Step 4: Metadata Embedding & Folder Routing**
   * Stamps all discovered labels into a single EXIF field (`Keywords`).
   * Evaluates a deterministic routing rulebook to move the physical files into your preferred chronological or thematic directories.

---

## 2. Technical Model Specifications

To stay safely under the **250,000 weight budget** for the custom classification CNN, and under 2 Million parameters for the face networks, the models leverage specialized structural efficiency patterns:

* **Depthwise Separable Convolutions:** Standard convolutions are replaced with paired Depthwise (spatial filtering per channel) and Pointwise (1 × 1 channel mixing) convolutions. This single change eliminates up to 90% of the mathematical floating-point operations (FLOPs).
* **Global Average Pooling (GAP):** To avoid the massive parameter explosion caused by flattening a deep convolutional feature map into a classic dense layer, a GAP layer takes the average of each spatial channel down to a simple 1 × 1 × Channels array immediately preceding the final linear layer.
* **Aggressive Input Downsampling:** Images are aggressively resized to 128 × 128 or 96 × 96 pixels prior to CNN evaluation, removing roughly 70% of the input data load compared to standard 224 × 224 training shapes.

### Optimized Network Footprint
| Task / Component | Architecture Model | Weight / Parameter Count | Approx. Disk Size | Hardware Load Profile |
| :--- | :--- | :--- | :--- | :--- |
| **Face Detection** | Ultra-Light Generic Face Detector | ~400k | ~1.1 MB | Extremely Low (Fast Pass) |
| **Face Identification** | MobileFaceNet / GhostFaceNet | ~1.2M | ~4.5 MB | Medium (Conditional Only) |
| **General Category Sorter**| Custom Multi-Label CNN + GAP | ~160k | ~1.5 MB | Exceptionally Low |

---

## 3. Ground Truth & Multi-Label Training Setup

To train the custom classification network using your baseline dataset of 4,000 images without a CSV file, you arrange your training images into descriptive folder combinations.

### Folder Layout Strategy
Sort your training images into directories representing either single categories or target combinations. Your PyTorch script can look at the parent directory name to automatically assign the multi-hot target vectors:

```text
dataset/
├── scenery/          --> Assigned [0, 0, 1]
├── cars/             --> Assigned [1, 0, 0]
├── people/           --> Assigned [0, 1, 0]
├── cars_and_people/  --> Assigned [1, 1, 0]
└── misc/             --> Assigned [0, 0, 0]
```

### Training Mathematics & Optimization
* **Loss Function:** `nn.BCEWithLogitsLoss()` (Binary Cross-Entropy with Logits). This forces the network to calculate losses for each of the 3 outputs independently, allowing an image in the `cars_and_people` folder to train both the car and person neurons simultaneously.
* **Regularization:** Heavy data augmentation (Random Horizontal Flips, Random Rotations, Color Jitter) and an aggressive `Dropout(0.3)` layer are appended right before the classification boundary to prevent overfitting on the 4,000 sample set.

---

## 4. Hardware Optimization & Ecosystem Integration

### Compiling for Legacy CPUs
Because a 1st-gen i7 processor lacks AVX/AVX2 vector acceleration, traditional Python-bound deep learning runtimes (PyTorch/TensorFlow) introduce severe overhead.
* **The Solution:** Models are exported directly from training into the **ONNX (Open Neural Network Exchange)** format.
* **Execution:** Running via `onnxruntime` inside optimized C++ or lightweight Python layers allows the system to utilize highly-optimized `SSE4.1/4.2` SIMD math operations, speeding up inference 5x to 10x over native Python frameworks.

### Zero-Friction Viewer Layer (PhotoPrism)
Instead of engineering a complex, vulnerable, custom photo gallery web application from scratch, the pipeline uses **PhotoPrism** as the frontend interface.

```
[ Incoming Images ] 
       │
       ▼
[ Your Custom Script ]  ──> Executes ONNX Pipeline (Cascading Models)
       │                ──> Appends Discovery Tags to EXIF Metadata
       ▼
[ Organized NAS Folders ] 
       │
       ▼
[ PhotoPrism (Read-Only) ] ──> Indexes EXIF Tags Natively
                           ──> Populates Search Index and Maps
                           ──> Handles "Secret Links" Accountless Sharing
```

* **Read-Only Mode:** PhotoPrism is mounted to the NAS storage directory with the environment flag `PHOTOPRISM_READONLY="true"`. It will never move, rename, or touch your folder structure.
* **EXIF Consumption:** Your sorting script writes the final predictions (e.g., `["John", "Cars"]`) straight into the image's standard metadata header under the `Keywords` tag. PhotoPrism reads these tags on ingestion, making them instantly searchable.
* **Secure Accountless Sharing:** PhotoPrism's **Secret Links** mechanism enables generating cryptographic web tokens on an album or folder level. Recipients can securely view, map, and download photos without requiring an account, and links can be configured with discrete expiration rules.
