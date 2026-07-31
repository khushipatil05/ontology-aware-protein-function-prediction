# ontology-aware-protein-function-prediction

The goal is to predict a protein's biological function directly from its amino acid sequence, addressing the huge gap 
between known sequences and experimentally verified functions. The design uses a pretrained ESM-2 protein 
language model as a frozen feature extractor — so we don't have to fine-tune a huge transformer — feeding 
embeddings into a multi-label MLP classifier that predicts Gene Ontology (GO) terms across three categories: 
Molecular Function, Biological Process, and Cellular Component. The key differentiator is that it's ontology-aware — 
it's designed to respect GO's parent-child hierarchy so the model doesn't predict a child term without its parent, 
which is a common biological-consistency failure in simpler models.

## Status

🚧 Design and Planning Phase

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
