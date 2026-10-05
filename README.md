# 🧬 Ontology-Aware Protein Function Prediction

## From Primary Protein Sequence Using Frozen Protein Language Models

An AI-driven bioinformatics project for predicting protein functions directly from amino acid sequences using frozen ESM-2 protein language model embeddings, a multi-label MLP classifier, and Gene Ontology (GO) hierarchy-aware learning.

## 📌 Overview

Protein function annotation is a fundamental task in computational biology with applications in drug discovery, disease analysis, genomics, and biotechnology.

The rapid growth of protein sequence databases has created a significant gap between the number of known protein sequences and the number of proteins whose functions have been experimentally validated.

This project proposes an automated deep learning framework that predicts protein functions directly from their primary amino acid sequences.

The system uses:

ESM-2 as a frozen protein language model and feature extractor
Multi-Layer Perceptron (MLP) for multi-label classification
Gene Ontology (GO) for functional annotation
GO hierarchy-aware learning to improve biological consistency
Precision, Recall, F1-score, Fmax, and AUPR for evaluation
Streamlit for the final user-facing application

The proposed system predicts GO terms across the three major Gene Ontology categories:
🧪 Molecular Function (MF)
🧬 Biological Process (BP)
🧫 Cellular Component (CC)

## 🎯 Problem Statement

The number of known protein sequences is increasing rapidly due to advances in sequencing technologies, while experimentally verified protein functions cannot keep pace.

Traditional experimental annotation is:
Time-consuming
Expensive
Resource-intensive
Difficult to scale

Existing computational approaches may also struggle to capture complex sequence patterns, handle severe class imbalance, and maintain the hierarchical relationships between Gene Ontology terms.

Therefore, there is a need for an automated, scalable, accurate, and biologically consistent protein function prediction system capable of analyzing large volumes of protein sequence data.

## 💡 Objectives

The project aims to:

1. Develop an automated deep learning system for predicting protein functions directly from amino acid sequences using pretrained ESM-2 embeddings.
2. Design a multi-label classification model for predicting Gene Ontology terms.
3. Incorporate Gene Ontology hierarchical relationships to produce biologically consistent predictions.
4. Evaluate the proposed model using appropriate metrics for imbalanced multi-label classification.
5. Develop a user-friendly platform that allows users to submit protein sequences and obtain predicted functional annotations.

## 🔬 Methodology

### 1. Data Collection

Protein sequences and functional annotations are collected from publicly available biological resources:
1. UniProt Swiss-Prot
2. Gene Ontology Consortium
3. UniProt-GOA

The project uses curated protein sequences and functional annotations for model development.

### 2. Data Preprocessing

The preprocessing stage includes:
1. Removing duplicate protein sequences
2. Filtering noisy annotations
3. Selecting relevant experimentally supported labels
4. Mapping proteins to their associated GO terms
5. Converting GO annotations into multi-hot encoded labels

Preparing Gene Ontology hierarchy information

### 3. Feature Extraction Using ESM-2

The project uses a pretrained ESM-2 protein language model to generate numerical representations of protein sequences.
Instead of fine-tuning the entire transformer, ESM-2 is used as a frozen feature extractor.

This approach reduces computational requirements while taking advantage of the rich biological representations learned by the pretrained protein language model.

### 4. Multi-Label Classification

A protein can have multiple biological functions simultaneously.

Therefore, protein function prediction is formulated as a multi-label classification problem.

The extracted ESM-2 embeddings are passed to a 3-layer Multi-Layer Perceptron (MLP) that predicts multiple GO terms for each protein.

### 5. Ontology-Aware Learning

Gene Ontology is organized as a hierarchical structure consisting of relationships between broader parent terms and more specific child terms.

A major objective of this project is to incorporate this hierarchy into the prediction process.

The proposed hierarchical loss mechanism encourages the model to produce biologically consistent predictions and reduces cases where a specific child GO term is predicted without its corresponding broader parent annotation.

### 6. Dataset

UniProt Swiss-Prot - Curated protein sequences and expert-reviewed annotations
Gene Ontology (GO) - Functional terms and hierarchical relationships
UniProt-GOA - Protein-GO functional associations

Exact dataset sizes and final train/validation/test splits will be documented after the complete preprocessing pipeline and experimental setup are finalized.

## 🧪 Why Frozen ESM-2?

Large protein language models contain millions of learned parameters and can be computationally expensive to fine-tune.

This project therefore uses ESM-2 as a frozen model.

Advantages
1. Reduces computational requirements
2. Avoids expensive transformer fine-tuning
3. Allows reusable protein representations
4. Simplifies downstream model training
5. Makes the approach more practical for limited computational environments

## Team Members

- Khushi Patil
- Diya Kalghatgi
- Nidhi Nayak

## Guide

Prof. AshaRani Patil

## Progress

- [x] Literature Survey
- [x] Research Gap
- [x] Problem Statement
- [x] Objectives
- [x] Proposed Methodology
- [x] Data Collection
- [ ] Model Development
- [ ] Evaluation
- [ ] Deployment
