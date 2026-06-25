<!-- SHIELDS -->
<div align="left">

![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.1%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
[![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-cutting?label=License)](LICENSE.txt)

# Qiskit-QCBMs

## Quantum Circuit Born Machines for Molecular Data Augmentation in Qiskit

Quantum Circuit Born Machines (QCBMs) are quantum generative models that learn probability distributions using parameterized quantum circuits. This project extends the QCBM framework toward **molecular data augmentation**, combining **STONED-SELFIES**, **classical generative models**, **graph-based molecular encodings**, and **hardware-aware quantum machine learning** for the exploration of chemical space around known **KRAS inhibitors**.

---

# Table of Contents

- [About](#about)
- [Project Objectives](#project-objectives)
- [Workflow Overview](#workflow-overview)
- [Repository Structure](#repository-structure)
- [Pipeline Modules](#pipeline-modules)
- [Experimental Notebooks](#experimental-notebooks)
- [Documentation](#documentation)
- [Installation](#installation)
- [Deprecation Policy](#deprecation-policy)
- [References](#references)
- [License](#license)

---

# About

Quantum Circuit Born Machines represent probability distributions using parameterized quantum circuits trained via classical optimization.

Compared with classical probabilistic models, QCBMs offer

- highly expressive probability distributions
- compact quantum representations
- suitability for noisy intermediate-scale quantum (NISQ) hardware
- natural integration with hybrid quantum-classical optimization

This repository applies QCBMs to molecular generation and augmentation by combining them with molecular encodings, clustering techniques, graph neural networks, latent-space learning, reinforcement learning, and hardware execution.

---

# Project Objectives

The project aims to build an end-to-end molecular generation framework capable of

- generating chemically valid molecules
- augmenting small molecular datasets
- exploring neighborhoods of known KRAS inhibitors
- benchmarking quantum generative models against classical baselines
- evaluating execution on both simulators and real quantum hardware

---

# STONED-SELFIES Molecular Augmentation

The molecular augmentation stage builds upon ideas from the **STONED-SELFIES** algorithm.

Original implementation inspiration:

https://github.com/aspuru-guzik-group/quantum-generative-models

especially

```
stoned_algorithm/stoned_sim.py
```

Additional references:

- https://www.nature.com/articles/s41587-024-02526-3
- https://arxiv.org/abs/2402.08210

---

## Dataset Expansion

Initial dataset

```
643 molecules
```

Augmented dataset

```
6430 molecules
```

Each original molecule generates

```
N = 9
```

valid neighboring molecular mutations.

For large-scale workflows similar to those presented in the referenced work,

```
N ≈ 1000
```

can be used.

The generated datasets contain

- SMILES
- SELFIES

representations.

Additional molecular representations include

- Molecular fingerprints
- Molecular descriptors
- Graph representations
- Latent embeddings

which can be used for downstream machine learning.

---

# Workflow Overview

```
Original Molecules
        │
        ▼
STONED-SELFIES Augmentation
        │
        ▼
Fingerprint / Descriptor Generation
        │
        ▼
Multiple Molecular Encodings
        │
        ▼
Quantum Circuit Born Machine Training
        │
        ▼
Classical Baselines
        │
        ▼
Hardware Execution
        │
        ▼
Evaluation Metrics
        │
        ▼
Benchmark Comparison
```

---

# Repository Structure

## Augmentation

```
augmentation/
├── generator.py
└── retrieval.py
```

Responsible for

- molecular retrieval
- STONED generation
- augmentation pipeline

---

## Classical Baselines

```
baselines/
├── classical_born.py
├── markov_born.py
└── stoned.py
```

Includes

- STONED baseline
- Classical Born Machine
- Markov Born Machine

---

## Quantum Circuit Born Machine

```
qcbm/
├── circuit.py
├── losses.py
├── trainer.py
├── sampler.py
└── qcbm.py
```

Implements

- parameterized quantum circuits
- training algorithms
- sampling
- loss computation

---

## Quantum Hardware Support

```
hardware/
├── backend_selection.py
├── topology.py
├── hardware_circuit.py
├── hardware_sampler.py
└── hardware_results.py
```

Supports

- IBM Quantum backend selection
- topology-aware transpilation
- hardware sampling
- execution analysis

---

# Advanced Molecular Pipeline

## Configuration

```
src/molaug_qcbm/config.py
```

Project configuration.

---

## Main Pipeline

```
src/molaug_qcbm/pipeline.py
```

End-to-end execution pipeline.

---

## Benchmark

```
src/molaug_qcbm/benchmark.py
```

Performance comparison across all implemented models.

---

# Evaluation Modules

```
evaluation/
├── validity.py
├── uniqueness.py
├── novelty.py
├── qed.py
└── sa_score.py
```

Metrics include

- Validity
- Uniqueness
- Novelty
- QED
- Synthetic Accessibility Score

---

# Molecular Preprocessing

```
preprocessing/
├── fingerprints.py
└── descriptors.py
```

Generates

- molecular fingerprints
- physicochemical descriptors

---

# Molecular Encodings

```
encoding/
├── chow_liu.py
├── graph_builder.py
├── gnn_embedding.py
├── scaffold_encoding.py
├── fragment_encoding.py
├── gmm_encoding.py
└── autoencoder_latent.py
```

Supported representations

- Chow-Liu Trees
- Molecular Graphs
- Graph Neural Network embeddings
- Molecular Scaffolds
- Molecular Fragments
- Gaussian Mixture Models
- Autoencoder latent vectors

---

# Proposed Quantum Models

## Chow-Liu QCBM

```
proposed/chowliu_qcbm.py
```

Combines Chow-Liu probabilistic encoding with QCBMs.

---

## KMeans QCBM

```
proposed/kmeans_qcbm.py
```

Clusters molecules prior to quantum training.

---

## GNN QCBM

```
proposed/gnn_qcbm.py
```

Uses graph neural network embeddings as QCBM inputs.

---

## Latent QCBM

```
proposed/latent_qcbm.py
```

Learns latent molecular distributions.

---

## Reinforcement Learning QCBM

```
proposed/rl_qcbm.py
```

Integrates reinforcement learning with quantum generation.

---

# Reconstruction

```
augmentation/reconstruction.py
```

Reconstructs molecular structures from latent representations.

---

# Classical Deep Learning Baselines

```
baselines/
├── vae.py
├── gan.py
├── qvae.py
├── qgan.py
└── rl_generator.py
```

Includes comparisons with

- Variational Autoencoders
- Generative Adversarial Networks
- Quantum VAEs
- Quantum GANs
- Reinforcement Learning generators

---

# Big Data Support

```
bigdata/
├── parquet_loader.py
├── distributed_clustering.py
└── feature_store.py
```

Large-scale data processing using

- Parquet datasets
- distributed clustering
- feature storage

---

# Distributed Computing

```
bigdata/
├── spark_pipeline.py
└── dask_pipeline.py
```

Supports

- Apache Spark
- Dask

for scalable molecular processing.

---

# Complete Experimental Notebook Pipeline

```
00_bigdata_processing
        │
        ▼
01_data_exploration
        │
        ▼
02_classical_baselines
        │
        ▼
03_QCBM_statevector
        │
        ▼
04_QCBM_hardware
        │
        ▼
05_ChowLiu_QCBM
        │
        ▼
06_KMeans_QCBM
        │
        ▼
07_GNN_QCBM
        │
        ▼
08_Latent_QCBM
        │
        ▼
09_RL_QCBM
        │
        ▼
10_Benchmark
```

Each notebook progressively introduces increasingly sophisticated molecular representations and quantum generative models, culminating in a comprehensive benchmark across classical, hybrid, and quantum approaches.

---

# Documentation

Project documentation is available at

https://qiskit-qcbms.readthedocs.io/en/latest/

---

# Installation

Clone the repository

```bash
git clone <repository-url>
```

Install the package

```bash
pip install .
```

The package appears as

```
qiskit-addon-qcbm
```

Using `pip` installation is recommended whenever possible.

---

# Deprecation Policy

The project follows **Semantic Versioning (SemVer)** and adheres to the principles of the **Qiskit Deprecation Policy**.

Whenever feasible, deprecated interfaces remain available until suitable replacements have matured. Breaking changes are documented in release notes.

---

# References

**[1]**

Liu, Jin-Guo and Wang, Lei.

**Differentiable Learning of Quantum Circuit Born Machines**

Physical Review A, Volume 98, Issue 6, 2018.

https://journals.aps.org/pra/pdf/10.1103/PhysRevA.98.062324

---

**[2]**

STONED-SELFIES Molecular Generation

https://github.com/aspuru-guzik-group/quantum-generative-models

---

**[3]**

Nature Biotechnology

https://www.nature.com/articles/s41587-024-02526-3

---

**[4]**

ArXiv Version

https://arxiv.org/abs/2402.08210

---

# License

Apache License 2.0

See

```
LICENSE.txt
```