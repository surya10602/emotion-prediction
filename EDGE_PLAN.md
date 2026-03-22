# Edge Deployment & Optimization Plan
## Overview
Mental health and emotional journaling data is highly sensitive. Relying on cloud-based LLMs introduces privacy risks, API latency, and requires an active internet connection. To align with the goal of building user-centric, real-world systems, this pipeline was architected specifically for local, on-device execution (Edge AI).

Below is the strategy for deploying this hybrid system (Transformer + XGBoost + Rule Logic) directly to iOS and Android devices.

## 1. Deployment Approach
The system will run entirely on the client's device using a cross-platform execution engine.

- **Execution Environment:** We will utilize ONNX Runtime (ORT) for mobile. ONNX (Open Neural Network Exchange) provides a unified runtime that supports hardware acceleration on both iOS (via CoreML/Apple Neural Engine) and Android (via NNAPI/Snapdragon Hexagon).

- **Pipeline Translation:** Both the sentence-transformers model (all-MiniLM-L6-v2) and the trained XGBoost models will be exported to the .onnx format. The Python decision engine will be rewritten into native Swift (iOS) and Kotlin (Android) or compiled via a lightweight C++ wrapper to interface with the ONNX models.

## 2. Model Optimizations
To ensure the application does not drain battery life or consume excessive device storage, aggressive optimization techniques will be applied prior to deployment.

- **INT8 Quantization:** The weights of the MiniLM embedding model will be quantized from 32-bit floating-point (FP32) to 8-bit integers (INT8).

- **Pruning:** For the XGBoost models, we will limit the maximum depth (e.g., max_depth=4) and strictly control the number of estimators during training to prevent tree bloat.

## 3. Model Size & Latency Projections
- **Storage Footprint:**

  - **Base Model Size:** The standard all-MiniLM-L6-v2 is roughly 80MB.

  - **Optimized Size:** After INT8 quantization, the model size drops to ~22MB.

  - **Tabular Models:** The quantized XGBoost models combined will take up less than 1MB.

  - **Total Footprint:** ~23MB. This comfortably fits into the RAM of modern mobile devices without triggering OS memory warnings.

- **Latency:** Generating a 384-dimensional embedding from a short journal entry and passing it through a shallow XGBoost tree requires minimal compute. On a standard modern mobile NPU, total inference time (Understanding → Decision) will be under 50 milliseconds.

  - **Network Latency:** 0 ms (Offline execution).

## 4. Tradeoffs & Product Considerations
- **Advantages**
  - **Absolute Privacy:** User data (stress, sleep, intimate journal entries) never leaves the device. This is a massive product differentiator for a wellness app.

  - **Offline Availability:** The user can log their state and receive guidance from a mountaintop, an airplane, or a subway with no internet connection.

  - **Zero API Costs:** Scaling from 1,000 to 1,000,000 users costs exactly the same in server inference (which is $0).

- **Disadvantages & Mitigation Strategy**

   **1. Model Drift & Updates:**

    - **The Problem: Unlike a cloud API that can be updated silently, edge models are static until the user downloads an app update.**

    - **Mitigation:** We will implement a lightweight telemetry system (opt-in) that sends anonymized, aggregated error flags (e.g., instances where uncertain_flag == 1) back to our servers to inform the next major model release.

   **2. Context Window Limitations:**

    - **The Problem:** MiniLM struggles with extremely long, rambling text compared to massive LLMs.

    - **Mitigation:** The UI will encourage concise reflections (e.g., a visual character limit or guided prompt) to keep inputs within the model's optimal semantic window.

5. Handling Edge Cases On-Device (Robustness)
Real-world systems break gracefully. On mobile:

- **Missing Hardware Data:** If the user denies HealthKit/Google Fit permissions (resulting in missing sleep/stress metadata), the native application will default to passing median historical values to the model, ensuring the pipeline never crashes due to a null variable.

- **Micro-Inputs:** If a user types very short text (e.g., "fine"), the decision engine is weighted to trust the physiological metadata (if available) over the noisy text prediction.
